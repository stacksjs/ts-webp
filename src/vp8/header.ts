import type { VP8FrameHeader } from '../types'
import { BoolDecoder } from './bool-decoder'
import {
  AC_QUANT,
  COEF_UPDATE_PROBS,
  DC_QUANT,
  DEFAULT_COEF_PROBS,
  NUM_BLOCK_TYPES,
  NUM_COEF_BANDS,
  clampQi,
} from './tables'

/**
 * VP8 keyframe header parser — 1:1 port of libwebp's `VP8GetHeaders`,
 * `ParseSegmentHeader`, `ParseFilterHeader`, `ParsePartitions`,
 * `VP8ParseQuant`, and `VP8ParseProba` (src/dec/vp8_dec.c §9 + §13.5).
 *
 * Field order for a keyframe:
 *
 *   1. Frame tag (3 bytes uncompressed): keyframe?, profile,
 *      show_frame?, partition_length.
 *   2. Start code (3 bytes uncompressed): 0x9D 0x01 0x2A.
 *   3. Width/height + scales (4 bytes uncompressed).
 *   4. Bool-coded first partition begins:
 *      a. color_space (1 bit)
 *      b. clamp_type (1 bit)
 *      c. Segment header (variable)
 *      d. Filter header (variable, includes simple/level/sharpness/lf-delta)
 *      e. Partition table (2 bits log2_nparts + size headers if >1)
 *      f. Quantiser (7 bits + 5 deltas)
 *      g. update_proba bit (consumed and ignored on keyframes)
 *      h. Coefficient-probability update loop (4×8×3×11 entries)
 *      i. mb_no_skip_coef (1 bit) + prob_skip_false (8 bits if set)
 *
 * After this, mode info begins on the same first-partition bool decoder
 * (one MB at a time, `VP8ParseIntraModeRow`).
 */

/** A single segment's quant/filter override. */
export interface VP8Segment {
  /** Quantiser override (signed, applied via `absoluteDelta`). */
  quantiser: number
  /** Filter strength override (signed, applied via `absoluteDelta`). */
  filterStrength: number
}

export interface VP8SegmentHeader {
  useSegment: boolean
  updateMap: boolean
  /** When true, segment values replace the base; when false, they're added to it. */
  absoluteDelta: boolean
  /** Per-segment quantiser/filter overrides (NUM_MB_SEGMENTS = 4 entries). */
  segments: VP8Segment[]
  /** Three probabilities used when decoding per-MB segment ids. */
  segmentTreeProbs: Uint8Array
}

export interface VP8FilterHeader {
  /** True when filter_type == 1 (simple), false for the complex filter. */
  simple: boolean
  /** Base loop-filter level (post-segment override but pre-mode delta). */
  level: number
  sharpness: number
  /** True if `refLfDelta`/`modeLfDelta` should be applied. */
  useLfDelta: boolean
  /** Per-reference-frame loop-filter level deltas (4 entries). */
  refLfDelta: Int8Array
  /** Per-mode loop-filter level deltas (4 entries). */
  modeLfDelta: Int8Array
  /** Effective filter type: 0 (off), 1 (simple), 2 (complex). */
  filterType: 0 | 1 | 2
}

/** Per-segment dequantisation matrices, populated by `VP8ParseQuant`. */
export interface VP8QuantMatrix {
  /** [DC, AC] for Y luma blocks. */
  y1: [number, number]
  /** [DC, AC] for the Y2 (WHT) block. */
  y2: [number, number]
  /** [DC, AC] for U/V chroma blocks. */
  uv: [number, number]
  /** Chroma quantiser used for dithering strength (we don't dither). */
  uvQuant: number
}

export interface ParsedVP8Header {
  frame: VP8FrameHeader
  /** Number of token partitions (1, 2, 4, or 8). */
  numPartitions: number
  /** Byte offset (inside the VP8 chunk) where the partition-size table starts. */
  partitionsOffset: number
  colorSpace: number
  clampType: number
  segmentation: VP8SegmentHeader
  filter: VP8FilterHeader
  /** Per-segment dequantisation matrices (NUM_MB_SEGMENTS = 4 entries). */
  quant: VP8QuantMatrix[]
  /** Bool decoder positioned at the macroblock-mode-info area. */
  bool: BoolDecoder
  /** Probability of the per-MB skip-coef flag, or null when it isn't coded. */
  probSkipFalse: number | null
  /** Active coefficient-probability table (with any updates applied). */
  coefProbs: Uint8Array
}

const NUM_MB_SEGMENTS = 4
const NUM_REF_LF_DELTAS = 4
const NUM_MODE_LF_DELTAS = 4
const MB_FEATURE_TREE_PROBS = 3

export function parseVP8Header(data: Uint8Array): ParsedVP8Header {
  if (data.length < 10) {
    throw new Error('VP8: chunk shorter than 10-byte frame header')
  }

  // ── Step 1-3: uncompressed frame tag + start code + dimensions ──────────
  const frameTag = data[0] | (data[1] << 8) | (data[2] << 16)
  const keyframe = (frameTag & 0x01) === 0
  if (!keyframe) {
    throw new Error('VP8: only keyframes are supported (WebP never embeds inter-frames)')
  }
  const profile = (frameTag >> 1) & 0x07
  if (profile > 3) {
    throw new Error(`VP8: unknown profile ${profile}`)
  }
  const showFrame = ((frameTag >> 4) & 0x01) === 1
  if (!showFrame) {
    throw new Error('VP8: frame is not displayable')
  }
  const firstPartSize = (frameTag >> 5) & 0x7FFFF

  if (data[3] !== 0x9D || data[4] !== 0x01 || data[5] !== 0x2A) {
    throw new Error('VP8: invalid keyframe start code')
  }

  const widthAndScale = data[6] | (data[7] << 8)
  const heightAndScale = data[8] | (data[9] << 8)
  const width = widthAndScale & 0x3FFF
  const height = heightAndScale & 0x3FFF
  const xScale = widthAndScale >> 14
  const yScale = heightAndScale >> 14
  if (width === 0 || height === 0) {
    throw new Error('VP8: zero dimension')
  }

  const partitionStart = 10
  if (partitionStart + firstPartSize > data.length) {
    throw new Error('VP8: declared first-partition size exceeds chunk length')
  }
  const bool = new BoolDecoder(data, partitionStart, firstPartSize)

  // ── Step 4a-b: colour space + clamp type ────────────────────────────────
  const colorSpace = bool.readBit(0x80)
  const clampType = bool.readBit(0x80)

  // ── Step 4c: Segment header ─────────────────────────────────────────────
  const segmentation = parseSegmentHeader(bool)

  // ── Step 4d: Filter header ──────────────────────────────────────────────
  const filter = parseFilterHeader(bool)

  // ── Step 4e: Partition layout ───────────────────────────────────────────
  const numPartitions = 1 << bool.readLiteral(2)
  // Token partitions follow the first partition's bytes; the size table
  // (if numPartitions > 1) is encoded raw at `partitionStart + firstPartSize`.
  const partitionsOffset = partitionStart + firstPartSize

  // ── Step 4f: Quantiser ─────────────────────────────────────────────────
  const quant = parseQuant(bool, segmentation)

  // ── Step 4g: update_proba (consumed and ignored — for keyframes the
  // saved set is always the defaults) ────────────────────────────────────
  bool.readBit(0x80)

  // ── Step 4h: Coefficient-probability update loop ───────────────────────
  const coefProbs = new Uint8Array(DEFAULT_COEF_PROBS)
  for (let i = 0; i < NUM_BLOCK_TYPES; i++) {
    for (let j = 0; j < NUM_COEF_BANDS; j++) {
      for (let k = 0; k < 3; k++) {
        for (let l = 0; l < 11; l++) {
          const off = i * NUM_COEF_BANDS * 3 * 11 + j * 3 * 11 + k * 11 + l
          if (bool.readBit(COEF_UPDATE_PROBS[off]) === 1) {
            coefProbs[off] = bool.readLiteral(8)
          }
        }
      }
    }
  }

  // ── Step 4i: mb_no_skip_coef + prob_skip_false ─────────────────────────
  let probSkipFalse: number | null = null
  if (bool.readBit(0x80) === 1) {
    probSkipFalse = bool.readLiteral(8)
  }

  return {
    frame: {
      keyframe: true,
      version: profile,
      showFrame,
      firstPartSize,
      width,
      height,
      xScale,
      yScale,
    },
    numPartitions,
    partitionsOffset,
    colorSpace,
    clampType,
    segmentation,
    filter,
    quant,
    bool,
    probSkipFalse,
    coefProbs,
  }
}

// ---------------------------------------------------------------------------
// Sub-parsers
// ---------------------------------------------------------------------------

function parseSegmentHeader(bool: BoolDecoder): VP8SegmentHeader {
  const segments: VP8Segment[] = []
  for (let i = 0; i < NUM_MB_SEGMENTS; i++) {
    segments.push({ quantiser: 0, filterStrength: 0 })
  }
  const segmentTreeProbs = new Uint8Array(MB_FEATURE_TREE_PROBS)
  segmentTreeProbs.fill(255)

  const useSegment = bool.readBit(0x80) === 1
  let updateMap = false
  let absoluteDelta = true
  if (useSegment) {
    updateMap = bool.readBit(0x80) === 1
    if (bool.readBit(0x80) === 1) {
      // update segment data
      absoluteDelta = bool.readBit(0x80) === 1
      for (let s = 0; s < NUM_MB_SEGMENTS; s++) {
        segments[s].quantiser = bool.readBit(0x80) === 1 ? bool.readSignedLiteral(7) : 0
      }
      for (let s = 0; s < NUM_MB_SEGMENTS; s++) {
        segments[s].filterStrength = bool.readBit(0x80) === 1 ? bool.readSignedLiteral(6) : 0
      }
    }
    if (updateMap) {
      for (let s = 0; s < MB_FEATURE_TREE_PROBS; s++) {
        segmentTreeProbs[s] = bool.readBit(0x80) === 1 ? bool.readLiteral(8) : 255
      }
    }
  }
  return {
    useSegment,
    updateMap,
    absoluteDelta,
    segments,
    segmentTreeProbs,
  }
}

function parseFilterHeader(bool: BoolDecoder): VP8FilterHeader {
  const simple = bool.readBit(0x80) === 1
  const level = bool.readLiteral(6)
  const sharpness = bool.readLiteral(3)
  const useLfDelta = bool.readBit(0x80) === 1
  const refLfDelta = new Int8Array(NUM_REF_LF_DELTAS)
  const modeLfDelta = new Int8Array(NUM_MODE_LF_DELTAS)
  if (useLfDelta) {
    if (bool.readBit(0x80) === 1) {
      // update lf-delta
      for (let i = 0; i < NUM_REF_LF_DELTAS; i++) {
        if (bool.readBit(0x80) === 1) refLfDelta[i] = bool.readSignedLiteral(6)
      }
      for (let i = 0; i < NUM_MODE_LF_DELTAS; i++) {
        if (bool.readBit(0x80) === 1) modeLfDelta[i] = bool.readSignedLiteral(6)
      }
    }
  }
  const filterType: 0 | 1 | 2 = level === 0 ? 0 : simple ? 1 : 2
  return { simple, level, sharpness, useLfDelta, refLfDelta, modeLfDelta, filterType }
}

function parseQuant(bool: BoolDecoder, seg: VP8SegmentHeader): VP8QuantMatrix[] {
  const baseQ0 = bool.readLiteral(7)
  const dqy1Dc = bool.readBit(0x80) === 1 ? bool.readSignedLiteral(4) : 0
  const dqy2Dc = bool.readBit(0x80) === 1 ? bool.readSignedLiteral(4) : 0
  const dqy2Ac = bool.readBit(0x80) === 1 ? bool.readSignedLiteral(4) : 0
  const dquvDc = bool.readBit(0x80) === 1 ? bool.readSignedLiteral(4) : 0
  const dquvAc = bool.readBit(0x80) === 1 ? bool.readSignedLiteral(4) : 0

  const quant: VP8QuantMatrix[] = []
  for (let i = 0; i < NUM_MB_SEGMENTS; i++) {
    let q: number
    if (seg.useSegment) {
      q = seg.segments[i].quantiser
      if (!seg.absoluteDelta) q += baseQ0
    }
    else {
      if (i > 0) {
        // libwebp: reuse segment 0's matrix for non-segmented frames
        quant.push({
          y1: [quant[0].y1[0], quant[0].y1[1]],
          y2: [quant[0].y2[0], quant[0].y2[1]],
          uv: [quant[0].uv[0], quant[0].uv[1]],
          uvQuant: quant[0].uvQuant,
        })
        continue
      }
      q = baseQ0
    }
    const y1Dc = DC_QUANT[clampQi(q + dqy1Dc)]
    const y1Ac = AC_QUANT[clampQi(q)]
    const y2Dc = DC_QUANT[clampQi(q + dqy2Dc)] * 2
    let y2Ac = (AC_QUANT[clampQi(q + dqy2Ac)] * 101581) >> 16
    if (y2Ac < 8) y2Ac = 8
    const uvDcIdx = q + dquvDc
    const uvDc = DC_QUANT[uvDcIdx < 0 ? 0 : uvDcIdx > 117 ? 117 : uvDcIdx]
    const uvAc = AC_QUANT[clampQi(q + dquvAc)]
    quant.push({ y1: [y1Dc, y1Ac], y2: [y2Dc, y2Ac], uv: [uvDc, uvAc], uvQuant: q + dquvAc })
  }
  return quant
}
