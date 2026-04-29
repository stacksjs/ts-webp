import type { WebpEncodeOptions, WebpImageData } from './types'
import { createRiffContainer } from './riff'
import { encodeVP8L } from './vp8l/encoder'

/**
 * Encode RGBA pixel data to WebP format
 */
export function encode(
  imageData: WebpImageData,
  options: WebpEncodeOptions = {},
): Uint8Array {
  const { lossless = true } = options

  if (lossless) {
    return encodeLossless(imageData, options)
  }

  return encodeLossy(imageData, options)
}

/**
 * Encode as lossless WebP (VP8L)
 */
function encodeLossless(imageData: WebpImageData, options: WebpEncodeOptions): Uint8Array {
  // Encode image data as VP8L
  const vp8lData = encodeVP8L(imageData, options)

  // Wrap in RIFF container
  return createRiffContainer([
    { fourCC: 'VP8L', data: vp8lData },
  ])
}

/**
 * Encode as lossy WebP (VP8) — *not implemented*. A full VP8 encoder is a
 * substantial port (DCT/WHT, intra prediction across 4×4 and 16×16 modes,
 * boolean arithmetic coding, dequantisation, loop filter) and is outside
 * the scope of this library for now. We transparently fall back to
 * lossless so callers that pass `lossless: false` still get *some* output
 * rather than a thrown error.
 */
function encodeLossy(imageData: WebpImageData, options: WebpEncodeOptions): Uint8Array {
  return encodeLossless(imageData, options)
}

/**
 * Create extended WebP with alpha channel
 */
export function encodeWithAlpha(
  imageData: WebpImageData,
  options: WebpEncodeOptions = {},
): Uint8Array {
  const { width, height, data } = imageData

  // Check if alpha is needed
  let hasNonOpaqueAlpha = false
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] < 255) {
      hasNonOpaqueAlpha = true
      break
    }
  }

  if (!hasNonOpaqueAlpha) {
    // No alpha needed, use simple format
    return encode(imageData, options)
  }

  // Use lossless with alpha (VP8L handles alpha internally)
  if (options.lossless !== false) {
    return encode({ ...imageData, hasAlpha: true }, { ...options, lossless: true })
  }

  // For lossy with alpha, need VP8X extended format
  // Extract alpha channel
  const alphaData = new Uint8Array(width * height)
  for (let i = 0; i < width * height; i++) {
    alphaData[i] = data[i * 4 + 3]
  }

  // Create VP8X chunk
  const vp8xData = new Uint8Array(10)
  vp8xData[0] = 0x10 // Has alpha flag

  // Width - 1 (24 bits)
  vp8xData[4] = (width - 1) & 0xFF
  vp8xData[5] = ((width - 1) >> 8) & 0xFF
  vp8xData[6] = ((width - 1) >> 16) & 0xFF

  // Height - 1 (24 bits)
  vp8xData[7] = (height - 1) & 0xFF
  vp8xData[8] = ((height - 1) >> 8) & 0xFF
  vp8xData[9] = ((height - 1) >> 16) & 0xFF

  // Create alpha chunk (uncompressed for simplicity)
  const alphChunk = new Uint8Array(1 + alphaData.length)
  alphChunk[0] = 0 // Uncompressed
  alphChunk.set(alphaData, 1)

  // Encode image without alpha as VP8L
  const vp8lData = encodeVP8L(imageData, options)

  return createRiffContainer([
    { fourCC: 'VP8X', data: vp8xData },
    { fourCC: 'ALPH', data: alphChunk },
    { fourCC: 'VP8L', data: vp8lData },
  ])
}
