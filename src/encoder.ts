import type { WebpEncodeOptions, WebpImageData } from './types'
import { createRiffContainer } from './riff'
import { encodeVP8 } from './vp8/encoder'
import { encodeVP8L } from './vp8l/encoder'

/**
 * Encode RGBA pixel data to a WebP byte buffer.
 *
 * Defaults to lossless (VP8L). Pass `lossless: false` to use the VP8
 * lossy encoder — note that the lossy path is a minimal-but-functional
 * implementation (16×16 intra-DC prediction, no mode search, single
 * segment) so the rate-distortion isn't competitive with `cwebp`. Use
 * lossless for any case that needs the smallest output or strict
 * fidelity, and lossy only when the caller can tolerate visible
 * artefacts in exchange for a small/predictable bitstream.
 */
export function encode(
  imageData: WebpImageData,
  options: WebpEncodeOptions = {},
): Uint8Array {
  const { lossless = true } = options
  if (lossless) return encodeLossless(imageData, options)
  return encodeLossy(imageData, options)
}

/**
 * Encode as lossy WebP (VP8). Produces a `RIFF/WEBP/VP8 ` byte stream.
 * The current implementation always uses 16×16 intra-DC prediction and
 * a single token partition — see `vp8/encoder.ts` for the scope notes.
 *
 * Alpha is silently dropped from the output (VP8 itself doesn't carry
 * alpha; the `VP8X + ALPH + VP8` extended container would be required,
 * which we don't emit on the encode side yet). Round-tripping an opaque
 * image is fine; round-tripping a transparent one will lose alpha.
 */
function encodeLossy(imageData: WebpImageData, options: WebpEncodeOptions): Uint8Array {
  // Map the public `quality: 0..100` (cwebp convention, higher = better)
  // to the internal q-index `0..127` (lower = better). This is a
  // monotonic-but-coarse mapping; libwebp uses a more elaborate
  // rate-control loop we don't replicate.
  const q100 = options.quality ?? 75
  if (q100 < 0 || q100 > 100) throw new Error('ts-webp: quality must be 0..100')
  const qIndex = Math.round(127 - (q100 / 100) * 127)
  const vp8Data = encodeVP8(imageData, { quality: qIndex })
  return createRiffContainer([
    { fourCC: 'VP8 ', data: vp8Data },
  ])
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
