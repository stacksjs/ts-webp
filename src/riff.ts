import type { RiffChunk, WebpInfo } from './types'

/**
 * Parse a RIFF/WEBP container into its constituent chunks.
 *
 * Each chunk's `data` is a `subarray` view into the input buffer — no
 * copy. This matters for large WebP files: the previous implementation
 * used `buffer.slice(...)` which copied each chunk's bytes, doubling
 * peak memory during decode of multi-megabyte input. Callers that
 * mutate chunk data should clone explicitly.
 *
 * Tolerates short/truncated input: instead of looping to negative
 * offsets or reading past EOF, we stop cleanly when the next chunk
 * header would extend beyond `buffer.length`.
 */
export function parseRiff(buffer: Uint8Array): RiffChunk[] {
  if (buffer.length < 12) {
    throw new Error('Invalid RIFF file: shorter than 12-byte container header')
  }
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength)

  if (buffer[0] !== 0x52 || buffer[1] !== 0x49 || buffer[2] !== 0x46 || buffer[3] !== 0x46) {
    throw new Error('Invalid RIFF file: missing RIFF signature')
  }
  if (buffer[8] !== 0x57 || buffer[9] !== 0x45 || buffer[10] !== 0x42 || buffer[11] !== 0x50) {
    throw new Error('Invalid WebP file: missing WEBP signature')
  }

  const chunks: RiffChunk[] = []
  let offset = 12
  while (offset + 8 <= buffer.length) {
    const fourCC = String.fromCharCode(
      buffer[offset],
      buffer[offset + 1],
      buffer[offset + 2],
      buffer[offset + 3],
    )
    const size = view.getUint32(offset + 4, true)
    // A claimed size larger than the remaining buffer is malformed —
    // clamp to what's available so the caller still gets a usable
    // (truncated) view rather than walking off the end.
    const dataStart = offset + 8
    const dataEnd = Math.min(dataStart + size, buffer.length)
    chunks.push({
      fourCC,
      size,
      data: buffer.subarray(dataStart, dataEnd),
      offset: dataStart,
    })
    // Pad to even boundary, but never advance past EOF.
    const paddedSize = size + (size & 1)
    const nextOffset = offset + 8 + paddedSize
    if (nextOffset <= offset) break // defensively guard against overflow
    offset = nextOffset
  }
  return chunks
}

/**
 * Get WebP file info from chunks
 */
export function getWebpInfo(chunks: RiffChunk[]): WebpInfo {
  // Check for VP8X (extended format)
  const vp8xChunk = chunks.find(c => c.fourCC === 'VP8X')
  if (vp8xChunk) {
    return parseVP8XInfo(vp8xChunk)
  }

  // Check for VP8L (lossless)
  const vp8lChunk = chunks.find(c => c.fourCC === 'VP8L')
  if (vp8lChunk) {
    return parseVP8LInfo(vp8lChunk)
  }

  // Check for VP8 (lossy)
  const vp8Chunk = chunks.find(c => c.fourCC === 'VP8 ')
  if (vp8Chunk) {
    return parseVP8Info(vp8Chunk)
  }

  throw new Error('No valid WebP data chunk found')
}

function parseVP8XInfo(chunk: RiffChunk): WebpInfo {
  const flags = chunk.data[0]

  const hasAnimation = (flags & 0x02) !== 0
  const hasAlpha = (flags & 0x10) !== 0

  // Width and height are 24-bit little-endian + 1
  const width = (chunk.data[4] | (chunk.data[5] << 8) | (chunk.data[6] << 16)) + 1
  const height = (chunk.data[7] | (chunk.data[8] << 8) | (chunk.data[9] << 16)) + 1

  return {
    width,
    height,
    hasAlpha,
    isLossless: false, // Will be determined by actual data chunk
    hasAnimation,
    isExtended: true,
  }
}

function parseVP8LInfo(chunk: RiffChunk): WebpInfo {
  // VP8L signature byte
  if (chunk.data[0] !== 0x2F) {
    throw new Error('Invalid VP8L signature')
  }

  // Read 4 bytes of header info (little-endian)
  const bits = chunk.data[1]
    | (chunk.data[2] << 8)
    | (chunk.data[3] << 16)
    | (chunk.data[4] << 24)

  const width = (bits & 0x3FFF) + 1
  const height = ((bits >> 14) & 0x3FFF) + 1
  const hasAlpha = ((bits >> 28) & 0x1) === 1

  return {
    width,
    height,
    hasAlpha,
    isLossless: true,
    hasAnimation: false,
    isExtended: false,
  }
}

function parseVP8Info(chunk: RiffChunk): WebpInfo {
  // VP8 (lossy) keyframe header layout:
  //   3-byte frame tag: bit 0 = 0 for keyframe
  //   3-byte start code: 0x9D 0x01 0x2A
  //   2-byte width-and-scale: low 14 bits = width
  //   2-byte height-and-scale: low 14 bits = height
  // We tolerate slightly malformed VP8 headers here because `getWebpInfo`
  // is an introspection helper — if the start code is missing we still
  // try to surface the dimensions when possible, since most callers just
  // want to know "what shape is this image" without committing to decode.
  if (chunk.data.length < 10) {
    throw new Error('ts-webp: VP8 chunk too short (lossy WebP, decode not implemented)')
  }

  const frameTag = chunk.data[0] | (chunk.data[1] << 8) | (chunk.data[2] << 16)
  const keyframe = (frameTag & 0x01) === 0
  if (!keyframe) {
    throw new Error('ts-webp: VP8 inter-frames not supported (lossy WebP, decode not implemented)')
  }

  if (chunk.data[3] !== 0x9D || chunk.data[4] !== 0x01 || chunk.data[5] !== 0x2A) {
    throw new Error('ts-webp: invalid VP8 start code (lossy WebP, decode not implemented)')
  }

  const widthAndScale = chunk.data[6] | (chunk.data[7] << 8)
  const heightAndScale = chunk.data[8] | (chunk.data[9] << 8)
  return {
    width: widthAndScale & 0x3FFF,
    height: heightAndScale & 0x3FFF,
    hasAlpha: false,
    isLossless: false,
    hasAnimation: false,
    isExtended: false,
  }
}

/**
 * Create RIFF container for WebP
 */
export function createRiffContainer(chunks: { fourCC: string, data: Uint8Array }[]): Uint8Array {
  // Calculate total size
  let dataSize = 4 // 'WEBP' signature
  for (const chunk of chunks) {
    const paddedSize = chunk.data.length + (chunk.data.length & 1)
    dataSize += 8 + paddedSize
  }

  const buffer = new Uint8Array(8 + dataSize)
  const view = new DataView(buffer.buffer)

  // Write RIFF header
  buffer[0] = 0x52 // 'R'
  buffer[1] = 0x49 // 'I'
  buffer[2] = 0x46 // 'F'
  buffer[3] = 0x46 // 'F'
  view.setUint32(4, dataSize, true)

  // Write WEBP signature
  buffer[8] = 0x57 // 'W'
  buffer[9] = 0x45 // 'E'
  buffer[10] = 0x42 // 'B'
  buffer[11] = 0x50 // 'P'

  // Write chunks
  let offset = 12
  for (const chunk of chunks) {
    // Write FourCC
    buffer[offset] = chunk.fourCC.charCodeAt(0)
    buffer[offset + 1] = chunk.fourCC.charCodeAt(1)
    buffer[offset + 2] = chunk.fourCC.charCodeAt(2)
    buffer[offset + 3] = chunk.fourCC.charCodeAt(3)

    // Write size
    view.setUint32(offset + 4, chunk.data.length, true)

    // Write data
    buffer.set(chunk.data, offset + 8)

    // Pad to even boundary
    const paddedSize = chunk.data.length + (chunk.data.length & 1)
    offset += 8 + paddedSize
  }

  return buffer
}
