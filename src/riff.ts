import type { RiffChunk, WebpInfo } from './types'

/**
 * Parse RIFF container
 */
export function parseRiff(buffer: Uint8Array): RiffChunk[] {
  const chunks: RiffChunk[] = []
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength)

  // Check RIFF signature
  const riff = String.fromCharCode(buffer[0], buffer[1], buffer[2], buffer[3])
  if (riff !== 'RIFF') {
    throw new Error('Invalid RIFF file: missing RIFF signature')
  }

  // Check WEBP signature
  const webp = String.fromCharCode(buffer[8], buffer[9], buffer[10], buffer[11])
  if (webp !== 'WEBP') {
    throw new Error('Invalid WebP file: missing WEBP signature')
  }

  // Parse chunks
  let offset = 12

  while (offset < buffer.length - 8) {
    const fourCC = String.fromCharCode(
      buffer[offset],
      buffer[offset + 1],
      buffer[offset + 2],
      buffer[offset + 3],
    )

    const size = view.getUint32(offset + 4, true)

    // Pad to even boundary
    const paddedSize = size + (size & 1)

    chunks.push({
      fourCC,
      size,
      data: buffer.slice(offset + 8, offset + 8 + size),
      offset: offset + 8,
    })

    offset += 8 + paddedSize
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
  // Parse VP8 frame header
  const frameTag = chunk.data[0] | (chunk.data[1] << 8) | (chunk.data[2] << 16)

  const keyframe = (frameTag & 0x01) === 0

  if (!keyframe) {
    throw new Error('VP8 inter-frames not supported')
  }

  // Check start code
  if (chunk.data[3] !== 0x9D || chunk.data[4] !== 0x01 || chunk.data[5] !== 0x2A) {
    throw new Error('Invalid VP8 start code')
  }

  // Read dimensions (16-bit little-endian with scale in upper bits)
  const widthAndScale = chunk.data[6] | (chunk.data[7] << 8)
  const heightAndScale = chunk.data[8] | (chunk.data[9] << 8)

  const width = widthAndScale & 0x3FFF
  const height = heightAndScale & 0x3FFF

  return {
    width,
    height,
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
