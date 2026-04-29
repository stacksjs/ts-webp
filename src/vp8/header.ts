import type { VP8FrameHeader } from '../types'
import { BoolDecoder } from './bool-decoder'

/**
 * VP8 keyframe header parser.
 *
 * The header occupies the first few bytes of the `VP8 ` chunk and
 * contains everything a decoder needs to set up its tables before
 * decoding macroblock coefficients. WebP only ever embeds keyframes
 * (intra frames) — there's no temporal prediction across frames.
 *
 * Layout (per RFC 6386 §9):
 *
 *   3 bytes: frame tag
 *     bit 0       = keyframe? (0 = keyframe in spec language; we assert)
 *     bits 1..3   = version
 *     bit 4       = show_frame
 *     bits 5..23  = first_partition_size
 *
 *   3 bytes: keyframe start code (0x9D 0x01 0x2A)
 *
 *   2 bytes: width-and-scale (LE)
 *     bits 0..13  = width in pixels
 *     bits 14..15 = horizontal scale
 *
 *   2 bytes: height-and-scale (LE)  — same layout
 *
 *   First partition (boolean-coded; size from the frame tag):
 *     1 bit  color_space (must be 0)
 *     1 bit  clamping_type
 *     1 bit  segmentation_enabled
 *       (if set: segmentation update header, ~50 bits)
 *     1 bit  filter_type
 *     6 bits loop_filter_level
 *     3 bits sharpness_level
 *     1 bit  loop_filter_adj_enabled
 *       (if set: loop filter delta header)
 *     2 bits log2(num_token_partitions)
 *     ...quantizer + entropy refresh + sign bias + last-frame-refresh...
 *
 * For WebP we treat every frame as a keyframe and ignore inter-frame
 * fields. The parser reads everything the decoder needs into a single
 * `ParsedHeader` record.
 */

/** Quantiser indices for Y, U/V chroma, and the Y2 (DC) coefficients. */
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
  /** Per-reference-frame loop-filter-level deltas (intra has 4 entries). */
  refDeltas: Int8Array
  /** Per-mode loop-filter-level deltas. */
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
}

/**
 * Read a VP8 frame header from a `VP8 ` chunk. Throws if the input
 * isn't a recognisable keyframe with the standard start code, or if
 * any field is out of the spec's allowed range.
 */
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

  // Frame header proper continues in the boolean-coded first partition.
  const partitionStart = 10
  if (partitionStart + firstPartSize > data.length) {
    throw new Error('VP8: declared first-partition size exceeds chunk length')
  }
  const bool = new BoolDecoder(data, partitionStart)

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

  // Quantiser indices follow; we only need the values, not the dequant
  // tables (which live in `tables.ts`).
  const yacQi = bool.readLiteral(7)
  const ydcDelta = readSignedDelta(bool)
  const y2dcDelta = readSignedDelta(bool)
  const y2acDelta = readSignedDelta(bool)
  const uvdcDelta = readSignedDelta(bool)
  const uvacDelta = readSignedDelta(bool)

  // Refresh flags (we don't track frame buffers; just consume them).
  const refreshGolden = bool.readLiteral(1)
  const refreshAltRef = bool.readLiteral(1)
  if (!refreshGolden) bool.readLiteral(2)
  if (!refreshAltRef) bool.readLiteral(2)
  bool.readLiteral(2) // sign bias golden + sign bias alt-ref
  bool.readLiteral(1) // refresh entropy probs
  bool.readLiteral(1) // refresh last

  // The compressed-header part ends here; mode-info + coefficients live
  // in the remaining bytes after `firstPartSize`.
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
  }
}

/**
 * Skip past the variable-length segmentation update header. The
 * decoder doesn't need the segmentation data unless segmentation is
 * actually used for a macroblock, which it isn't in any spec example.
 * For now we only read enough bits to keep the cursor correctly
 * positioned for what follows.
 */
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

/** Read an optional 4-bit signed delta (1 flag bit + 4 magnitude bits + sign). */
function readSignedDelta(bool: BoolDecoder): number {
  return bool.readLiteral(1) === 1 ? bool.readSignedLiteral(4) : 0
}
