import type { WebpDecodeOptions, WebpImageData } from './types'
import { parseRiff } from './riff'
import { decodeVP8 } from './vp8/decoder'
import { decodeVP8L } from './vp8l/decoder'

/**
 * Decode a WebP image buffer to RGBA pixel data.
 *
 * Currently supports the lossless (VP8L) format and the extended
 * (VP8X + VP8L) container that wraps it. Lossy (VP8) chunks throw a
 * clear `lossy not implemented` error rather than producing fake
 * output — see `vp8/decoder.ts` for context.
 */
export function decode(
  buffer: Uint8Array | ArrayBuffer,
  options: WebpDecodeOptions = {},
): WebpImageData {
  const data = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer
  const chunks = parseRiff(data)

  // Find the actual pixel-data chunk. VP8L is preferred over VP8 when
  // both are present (rare, but valid in extended format). We don't
  // need the VP8X header for anything other than container detection;
  // the VP8L bitstream carries its own width/height/alpha flag.
  const vp8lChunk = chunks.find(c => c.fourCC === 'VP8L')
  if (vp8lChunk) {
    const imageData = decodeVP8L(vp8lChunk.data)
    if (options.format === 'rgb') imageData.data = rgbaToRgb(imageData.data)
    return imageData
  }

  const vp8Chunk = chunks.find(c => c.fourCC === 'VP8 ')
  if (vp8Chunk) {
    // `decodeVP8` throws — see vp8/decoder.ts. We skip the RIFF
    // info-parser's VP8 frame-header validation here and let the
    // decoder produce its own clear "lossy not implemented" message,
    // matching what tests assert.
    return decodeVP8(vp8Chunk.data)
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
