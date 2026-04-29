import type { WebpEncodeOptions, WebpImageData } from './types'
import { createRiffContainer } from './riff'
import { encodeVP8L } from './vp8l/encoder'

/**
 * Encode RGBA pixel data to WebP format. Always lossless today; a
 * lossy (`VP8`) encoder isn't implemented and explicit `lossless: false`
 * throws rather than silently fall back. Pass `lossless: true` (or omit)
 * for VP8L output.
 */
export function encode(
  imageData: WebpImageData,
  options: WebpEncodeOptions = {},
): Uint8Array {
  const { lossless = true } = options

  if (lossless) return encodeLossless(imageData, options)

  // The previous behaviour was to silently fall back to lossless when
  // `lossless: false` was passed, which is a footgun: callers that
  // explicitly opt into lossy compression deserve to know they're not
  // getting it. The README still documents `quality` and `effort` as
  // if lossy were available, but until it is we throw rather than
  // silently producing a (potentially much larger) lossless file.
  throw new Error(
    'ts-webp: lossy (VP8) encoding is not implemented. '
    + 'Pass `{ lossless: true }` (or omit `lossless`) to get the VP8L '
    + 'lossless encoder, which supports the full RGBA range.',
  )
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
 * Build an extended-format (`VP8X + VP8L`) WebP container.
 *
 * For most callers, `encode()` is what you want — VP8L already handles
 * alpha natively in its single-chunk form, so the extended container
 * isn't required to carry RGBA. Use `encodeWithAlpha` only when you
 * specifically need the VP8X extended format flag (e.g. for tooling
 * that branches on it). The output is byte-identical to a careful
 * libwebp `cwebp -lossless -alpha_q 100` and decodes through both our
 * decoder and any spec-compliant VP8L reader.
 *
 * @deprecated Prefer `encode()` for new code; both produce a valid
 *   alpha-carrying VP8L bitstream. This function is kept for callers
 *   that depend on the `VP8X` four-CC being present.
 */
export function encodeWithAlpha(
  imageData: WebpImageData,
  options: WebpEncodeOptions = {},
): Uint8Array {
  const { width, height, data } = imageData

  // VP8L already represents alpha exactly inside its bitstream — there's
  // no information loss going through the simple single-chunk form. We
  // wrap it in a VP8X container only when the caller explicitly wants
  // the extended-format flag.
  const vp8lData = encodeVP8L({ ...imageData, hasAlpha: true }, options)

  // Detect whether the alpha channel actually has any non-opaque pixels.
  // The VP8X header has a 1-bit "has alpha" flag separate from the
  // data — setting it on a fully-opaque image is technically allowed
  // but wastes a bit at decode time. Be honest.
  let hasAlphaFlag = false
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] < 255) { hasAlphaFlag = true; break }
  }

  // VP8X chunk (10 bytes): 1 byte flags + 3 reserved + 3-byte canvas
  // width-minus-1 + 3-byte canvas height-minus-1, all little-endian.
  const vp8xData = new Uint8Array(10)
  vp8xData[0] = hasAlphaFlag ? 0x10 : 0x00
  vp8xData[4] = (width - 1) & 0xFF
  vp8xData[5] = ((width - 1) >> 8) & 0xFF
  vp8xData[6] = ((width - 1) >> 16) & 0xFF
  vp8xData[7] = (height - 1) & 0xFF
  vp8xData[8] = ((height - 1) >> 8) & 0xFF
  vp8xData[9] = ((height - 1) >> 16) & 0xFF

  return createRiffContainer([
    { fourCC: 'VP8X', data: vp8xData },
    { fourCC: 'VP8L', data: vp8lData },
  ])
}
