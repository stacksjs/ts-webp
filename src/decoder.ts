import type { WebpDecodeOptions, WebpImageData } from './types'
import { getWebpInfo, parseRiff } from './riff'
import { decodeVP8 } from './vp8/decoder'
import { decodeVP8L } from './vp8l/decoder'

/**
 * Decode a WebP image buffer to RGBA pixel data
 */
export function decode(
  buffer: Uint8Array | ArrayBuffer,
  options: WebpDecodeOptions = {},
): WebpImageData {
  const data = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer

  // Parse RIFF container
  const chunks = parseRiff(data)

  // Get image info
  const info = getWebpInfo(chunks)

  // Find and decode image data
  let imageData: WebpImageData

  // Check for VP8X (extended format)
  const vp8xChunk = chunks.find(c => c.fourCC === 'VP8X')

  if (vp8xChunk) {
    // Extended format - may have alpha
    const alphaChunk = chunks.find(c => c.fourCC === 'ALPH')
    const vp8lChunk = chunks.find(c => c.fourCC === 'VP8L')
    const vp8Chunk = chunks.find(c => c.fourCC === 'VP8 ')

    if (vp8lChunk) {
      imageData = decodeVP8L(vp8lChunk.data)
    }
    else if (vp8Chunk) {
      imageData = decodeVP8(vp8Chunk.data)

      // Apply alpha if present
      if (alphaChunk && info.hasAlpha) {
        applyAlphaChannel(imageData, alphaChunk.data)
      }
    }
    else {
      throw new Error('No valid image data found in extended WebP')
    }
  }
  else {
    // Simple format
    const vp8lChunk = chunks.find(c => c.fourCC === 'VP8L')
    const vp8Chunk = chunks.find(c => c.fourCC === 'VP8 ')

    if (vp8lChunk) {
      imageData = decodeVP8L(vp8lChunk.data)
    }
    else if (vp8Chunk) {
      imageData = decodeVP8(vp8Chunk.data)
    }
    else {
      throw new Error('No valid image data found')
    }
  }

  // Convert to RGB if requested
  if (options.format === 'rgb') {
    imageData.data = rgbaToRgb(imageData.data)
  }

  return imageData
}

/**
 * Splat an `ALPH`-chunk alpha channel into an already-decoded RGBA buffer.
 *
 * The ALPH chunk's first byte is a header (compression method + filter +
 * pre-processing flags); raw alpha samples start at offset 1. This was a
 * latent bug in the previous implementation, which copied from offset 0 —
 * but the dead code never ran because the lossy decoder always throws
 * before reaching here. Fixed in advance of any future VP8 lossy decode.
 */
function applyAlphaChannel(imageData: WebpImageData, alphaData: Uint8Array): void {
  const { data, width, height } = imageData
  const numPixels = width * height

  // Header byte at offset 0; only `compression_method == 0` (uncompressed)
  // is supported here. Compression methods 1..3 are VP8L-encoded alpha,
  // which would need decodeVP8L on the alpha-only payload.
  const compressionMethod = alphaData[0] & 0x03
  if (compressionMethod !== 0) {
    throw new Error(`ts-webp: ALPH compression method ${compressionMethod} not supported`)
  }

  const samples = alphaData.subarray(1)
  for (let i = 0; i < numPixels && i < samples.length; i++) {
    data[i * 4 + 3] = samples[i]
  }
  imageData.hasAlpha = true
}

function rgbaToRgb(rgba: Uint8Array): Uint8Array {
  const numPixels = rgba.length / 4
  const rgb = new Uint8Array(numPixels * 3)

  for (let i = 0; i < numPixels; i++) {
    rgb[i * 3] = rgba[i * 4]
    rgb[i * 3 + 1] = rgba[i * 4 + 1]
    rgb[i * 3 + 2] = rgba[i * 4 + 2]
  }

  return rgb
}
