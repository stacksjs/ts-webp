import type { WebpDecodeOptions, WebpImageData } from './types'
import { decodeAlpha } from './alpha'
import { parseRiff } from './riff'
import { decodeVP8 } from './vp8/decoder'
import { decodeVP8L } from './vp8l/decoder'

/**
 * Decode a WebP image buffer to RGBA pixel data.
 *
 * Supported containers:
 *   - simple `RIFF/WEBP/VP8 ` (lossy VP8) — bit-exact with `dwebp`
 *   - simple `RIFF/WEBP/VP8L` (lossless) — round-trips exactly
 *   - extended `RIFF/WEBP/VP8X + VP8L` (lossless with optional alpha)
 *   - extended `RIFF/WEBP/VP8X + ALPH + VP8 ` (lossy with alpha)
 *
 * VP8L is preferred over VP8 when both are present in the same
 * container (rare, but valid in some extended-format files).
 */
export function decode(
  buffer: Uint8Array | ArrayBuffer,
  options: WebpDecodeOptions = {},
): WebpImageData {
  const data = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer
  const chunks = parseRiff(data)

  const vp8lChunk = chunks.find(c => c.fourCC === 'VP8L')
  if (vp8lChunk) {
    const imageData = decodeVP8L(vp8lChunk.data)
    if (options.format === 'rgb') imageData.data = rgbaToRgb(imageData.data)
    return imageData
  }

  const vp8Chunk = chunks.find(c => c.fourCC === 'VP8 ')
  if (vp8Chunk) {
    const imageData = decodeVP8(vp8Chunk.data)
    // If a VP8X + ALPH + VP8 container carries an alpha plane, splice
    // it in over the all-0xFF alpha decodeVP8 emits. The ALPH chunk
    // appears BEFORE the VP8 chunk in canonical layout; we accept any
    // order so we tolerate non-canonical encoders.
    const alphChunk = chunks.find(c => c.fourCC === 'ALPH')
    if (alphChunk) {
      const alpha = decodeAlpha(alphChunk.data, imageData.width, imageData.height)
      const px = imageData.width * imageData.height
      for (let i = 0; i < px; i++) {
        imageData.data[i * 4 + 3] = alpha[i]
      }
      imageData.hasAlpha = true
    }
    if (options.format === 'rgb') imageData.data = rgbaToRgb(imageData.data)
    return imageData
  }

  throw new Error('ts-webp: no VP8 or VP8L chunk found in WebP container')
}

/**
 * Strip the alpha channel from an RGBA buffer. Caller passes
 * `format: 'rgb'` to opt into this; useful for callers that don't care
 * about transparency and would otherwise allocate 33 % more memory than
 * needed.
 */
function rgbaToRgb(rgba: Uint8Array): Uint8Array {
  const numPixels = rgba.length >>> 2
  const rgb = new Uint8Array(numPixels * 3)
  for (let i = 0; i < numPixels; i++) {
    const o = i * 4
    const p = i * 3
    rgb[p] = rgba[o]
    rgb[p + 1] = rgba[o + 1]
    rgb[p + 2] = rgba[o + 2]
  }
  return rgb
}
