import type { VP8FrameHeader } from '../types'
import { BoolDecoder } from './bool-decoder'
import { DEFAULT_COEF_PROBS, NUM_BLOCK_TYPES, NUM_COEF_BANDS } from './tables'

/**
 * VP8 keyframe header parser.
 *
 * For WebP we treat every frame as a keyframe (WebP never embeds
 * inter-frames). The parser reads the frame tag + start code + dimensions
 * uncompressed, then opens a `BoolDecoder` over the first partition and
 * consumes the bool-coded keyframe header per RFC 6386 §9.4-9.10.
 *
 * The order of bool-coded fields for a keyframe (per RFC 6386 §9.4ff,
 * cross-checked against libvpx's `vp8/decoder/decodemv.c`):
 *
 *   1.  color_space (1 bit) — must be 0
 *   2.  clamping_type (1 bit)
 *   3.  segmentation_enabled (1 bit)
 *       (if 1: full update_segmentation header)
 *   4.  filter_type (1 bit)
 *   5.  loop_filter_level (6 bits)
 *   6.  sharpness_level (3 bits)
 *   7.  mb_lf_adjustments (1 bit)
 *       (if 1: full mode_ref_lf_delta_update header)
 *   8.  log2_nparts (2 bits) — 1 << this is the token-partition count
 *   9.  y_ac_qi (7 bits) — base quantiser index
 *  10.  y_dc_delta, y2_dc_delta, y2_ac_delta, uv_dc_delta, uv_ac_delta
 *       (each: 1 flag bit + 5 magnitude+sign bits if set)
 *  11.  refresh_entropy_probs (1 bit)
 *  12.  COEF probability update loop (for each of the 4×8×3×11 entries:
 *       1 flag bit + 8 prob bits if flag set)
 *  13.  mb_no_skip_coef (1 bit)
 *       (if 1: prob_skip_false (8 bits))
 *
 * Inter-frame-only fields (refresh_golden/altref/sign_bias) are NOT
 * present for keyframes.
 */

export interface VP8Quantiser {
  yacQi: number
  ydcDelta: number
  y2dcDelta: number
  y2acDelta: number
  uvdcDelta: number
  uvacDelta: number
}

export interface VP8FilterParams {
  filterType: number
  level: number
  sharpness: number
  refDeltas: Int8Array
  modeDeltas: Int8Array
}

export interface ParsedVP8Header {
  frame: VP8FrameHeader
  numPartitions: number
  /** Byte offset (inside the VP8 chunk) where the residual partitions start. */
  partitionsOffset: number
  colorSpace: number
  clampType: number
  segmentation: { enabled: boolean }
  filter: VP8FilterParams
  quantiser: VP8Quantiser
  /** Boolean decoder positioned at the macroblock-mode-info area. */
  bool: BoolDecoder
  /** Probability skipped-block flag, or null if mb_no_skip_coef is 0. */
  probSkipFalse: number | null
  /** Active coefficient-probability table (with any updates applied). */
  coefProbs: Uint8Array
}

export function parseVP8Header(data: Uint8Array): ParsedVP8Header {
  if (data.length < 10) {
    throw new Error('VP8: chunk shorter than 10-byte frame header')
  }

  const frameTag = data[0] | (data[1] << 8) | (data[2] << 16)
  const keyframe = (frameTag & 0x01) === 0
  if (!keyframe) {
    throw new Error('VP8: only keyframes are supported (WebP never embeds inter-frames)')
  }
  const version = (frameTag >> 1) & 0x07
  const showFrame = ((frameTag >> 4) & 0x01) === 1
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

  const partitionStart = 10
  if (partitionStart + firstPartSize > data.length) {
    throw new Error('VP8: declared first-partition size exceeds chunk length')
  }
  // Bound the first-partition bool decoder by `firstPartSize` so the
  // header + mode-info reads can't accidentally pull bytes from the
  // following token partition. libvpx does the same via `buffer_end`.
  const bool = new BoolDecoder(data, partitionStart, firstPartSize)

  const colorSpace = bool.readLiteral(1)
  const clampType = bool.readLiteral(1)
  if (colorSpace !== 0) {
    throw new Error('VP8: unsupported colour space (must be 0 = YUV420)')
  }

  const segmentationEnabled = bool.readLiteral(1) === 1
  if (segmentationEnabled) {
    skipSegmentationHeader(bool)
  }

  const filterType = bool.readLiteral(1)
  const filterLevel = bool.readLiteral(6)
  const sharpness = bool.readLiteral(3)
  const refDeltas = new Int8Array(4)
  const modeDeltas = new Int8Array(4)
  const lfAdj = bool.readLiteral(1)
  if (lfAdj === 1) {
    const update = bool.readLiteral(1)
    if (update === 1) {
      for (let i = 0; i < 4; i++) {
        if (bool.readLiteral(1) === 1) refDeltas[i] = bool.readSignedLiteral(6)
      }
      for (let i = 0; i < 4; i++) {
        if (bool.readLiteral(1) === 1) modeDeltas[i] = bool.readSignedLiteral(6)
      }
    }
  }

  const numPartitions = 1 << bool.readLiteral(2)

  // Quantiser indices — y_ac_qi (7 bits) and 5 deltas (1 flag + 4 magnitude
  // + 1 sign each). Note: spec uses 4-bit magnitude, not 5.
  const yacQi = bool.readLiteral(7)
  const ydcDelta = readSignedDelta(bool)
  const y2dcDelta = readSignedDelta(bool)
  const y2acDelta = readSignedDelta(bool)
  const uvdcDelta = readSignedDelta(bool)
  const uvacDelta = readSignedDelta(bool)

  // Refresh entropy probs (keyframes still have this — RFC 6386 §9.10).
  // Followed by the coefficient-probability update loop.
  bool.readLiteral(1) // refresh_entropy_probs — for keyframes the
  // saved state always equals the defaults so this flag has no effect
  // on us; we still consume the bit.

  const coefProbs = new Uint8Array(DEFAULT_COEF_PROBS)
  // Update loop: 4 × 8 × 3 × 11 = 1056 entries, each with a flag bit +
  // optional 8-bit replacement probability. The decision probability is
  // a fixed table from the spec.
  for (let i = 0; i < NUM_BLOCK_TYPES; i++) {
    for (let j = 0; j < NUM_COEF_BANDS; j++) {
      for (let k = 0; k < 3; k++) {
        for (let l = 0; l < 11; l++) {
          if (bool.readBit(COEF_UPDATE_PROBS[i][j][k][l]) === 1) {
            coefProbs[i * NUM_COEF_BANDS * 3 * 11 + j * 3 * 11 + k * 11 + l] = bool.readLiteral(8)
          }
        }
      }
    }
  }

  // mb_no_skip_coef (always present on keyframes per RFC 6386 §9.11).
  let probSkipFalse: number | null = null
  if (bool.readLiteral(1) === 1) {
    probSkipFalse = bool.readLiteral(8)
  }

  // Token partitions follow `firstPartSize`.
  const partitionsOffset = partitionStart + firstPartSize

  return {
    frame: {
      keyframe: true,
      version,
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
    segmentation: { enabled: segmentationEnabled },
    filter: {
      filterType,
      level: filterLevel,
      sharpness,
      refDeltas,
      modeDeltas,
    },
    quantiser: {
      yacQi,
      ydcDelta,
      y2dcDelta,
      y2acDelta,
      uvdcDelta,
      uvacDelta,
    },
    bool,
    probSkipFalse,
    coefProbs,
  }
}

function skipSegmentationHeader(bool: BoolDecoder): void {
  const updateMap = bool.readLiteral(1)
  const updateData = bool.readLiteral(1)
  if (updateData === 1) {
    bool.readLiteral(1) // abs_delta
    for (let i = 0; i < 4; i++) {
      if (bool.readLiteral(1) === 1) bool.readSignedLiteral(7)
    }
    for (let i = 0; i < 4; i++) {
      if (bool.readLiteral(1) === 1) bool.readSignedLiteral(6)
    }
  }
  if (updateMap === 1) {
    for (let i = 0; i < 3; i++) {
      if (bool.readLiteral(1) === 1) bool.readLiteral(8)
    }
  }
}

/** Read an optional 4-bit signed delta (1 flag bit + 4 magnitude bits + sign bit). */
function readSignedDelta(bool: BoolDecoder): number {
  return bool.readLiteral(1) === 1 ? bool.readSignedLiteral(4) : 0
}

// ---------------------------------------------------------------------------
// Coef-probability update probabilities — RFC 6386 §13.5 ("k_coef_update_probs")
// ---------------------------------------------------------------------------
//
// 4 × 8 × 3 × 11 table mirroring DEFAULT_COEF_PROBS; the encoder uses
// these to decide whether to emit a probability update for each entry.
// In practice cwebp leaves these at 252 (very biased toward "no update")
// for almost every entry, so the inner loop reads 1056 zero-bits and
// moves on. We transcribe the spec's table verbatim.

// eslint-disable-next-line pickier/no-unused-vars
const COEF_UPDATE_PROBS: ReadonlyArray<ReadonlyArray<ReadonlyArray<ReadonlyArray<number>>>> = [
  [
    [
      [177, 155, 250, 247, 255, 255, 255, 255, 255, 255, 255],
      [250, 245, 254, 254, 254, 255, 255, 255, 255, 255, 255],
      [234, 246, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [251, 244, 254, 254, 255, 255, 255, 255, 255, 255, 255],
      [251, 251, 254, 252, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [254, 251, 254, 254, 255, 255, 255, 255, 255, 255, 255],
      [253, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [254, 252, 254, 255, 255, 255, 255, 255, 255, 255, 255],
      [249, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [254, 252, 254, 254, 255, 255, 255, 255, 255, 255, 255],
      [255, 252, 254, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [254, 252, 254, 255, 255, 255, 255, 255, 255, 255, 255],
      [253, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
  ],
  [
    [
      [186, 251, 250, 171, 192, 188, 201, 153, 251, 244, 255],
      [234, 251, 244, 254, 251, 252, 254, 252, 254, 252, 255],
      [251, 251, 252, 253, 254, 254, 251, 254, 252, 254, 255],
    ],
    [
      [255, 252, 254, 254, 255, 255, 255, 255, 255, 255, 255],
      [254, 251, 254, 254, 255, 255, 255, 255, 255, 255, 255],
      [253, 250, 252, 254, 254, 254, 255, 255, 255, 255, 255],
    ],
    [
      [255, 254, 253, 254, 254, 255, 255, 255, 255, 255, 255],
      [255, 252, 253, 254, 254, 255, 255, 255, 255, 255, 255],
      [255, 251, 253, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [253, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [253, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [253, 252, 254, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
      [254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
  ],
  [
    [
      [248, 255, 255, 228, 247, 255, 255, 255, 255, 255, 255],
      [255, 253, 255, 254, 254, 255, 255, 255, 255, 255, 255],
      [255, 250, 255, 254, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
  ],
  [
    [
      [248, 254, 249, 253, 255, 255, 230, 255, 255, 255, 255],
      [255, 253, 254, 254, 255, 255, 255, 255, 255, 255, 255],
      [244, 252, 255, 250, 254, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
      [254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
      [253, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
    [
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
      [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ],
  ],
]
