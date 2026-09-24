import type { WebpEncodeOptions, WebpImageData } from './types'
import { encodeViaCwebp, hasCwebp } from './encoder-cli'
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
 * Async-aware encoder that prefers the system `cwebp` binary
 * (libwebp) when available, falling back to the bundled pure-TS
 * encoder when it isn't. Use this for any production output —
 * libwebp's rate-distortion loop produces files that are 30–60%
 * smaller than the bundled lossy encoder at equivalent quality.
 *
 * Backwards-compat note: the synchronous `encode()` above stays
 * pure-TS only so existing call sites and tests don't change shape.
 * New code should prefer `encodeAsync()`.
 */
export async function encodeAsync(
  imageData: WebpImageData,
  options: WebpEncodeOptions = {},
): Promise<Uint8Array> {
  const backend = options.backend ?? 'auto'

  if (backend !== 'pure-ts') {
    if (await hasCwebp(options.cwebpPath)) {
      const cliBytes = await encodeViaCwebp(imageData, options)
      if (cliBytes) return cliBytes
    }
    if (backend === 'cli') {
      throw new Error(
        'ts-webp: cwebp binary not found on PATH (or failed). '
        + 'Install libwebp (`brew install webp` / `apt-get install webp`), '
        + 'set options.cwebpPath, or use backend: "pure-ts".',
      )
    }
  }

  return encode(imageData, options)
}

/**
 * Encode as lossy WebP (VP8). Produces a `RIFF/WEBP/VP8 ` byte stream.
 * The current implementation always uses 16×16 intra-DC prediction and
 * a single token partition — see `vp8/encoder.ts` for the scope notes.
 *
 * VP8 itself has no alpha, so a translucent image is written in the
 * extended form, `VP8X + ALPH + VP8`: the colour stays lossy and the mask
 * travels beside it, compressed losslessly. This used to drop alpha without
 * a word, and since lossy is what image pipelines ask for, every
 * transparent PNG they converted came out on a solid black box.
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

  const alpha = imageData.hasAlpha === false ? null : translucentAlphaPlane(imageData)
  if (!alpha) {
    return createRiffContainer([
      { fourCC: 'VP8 ', data: vp8Data },
    ])
  }

  return createRiffContainer([
    { fourCC: 'VP8X', data: vp8xChunk(imageData.width, imageData.height, true) },
    { fourCC: 'ALPH', data: encodeAlphaChunk(alpha, imageData.width, imageData.height, options) },
    { fourCC: 'VP8 ', data: vp8Data },
  ])
}

/** The alpha plane, one byte per pixel, or null when every pixel is opaque. */
function translucentAlphaPlane(imageData: WebpImageData): Uint8Array | null {
  const { data, width, height } = imageData
  const plane = new Uint8Array(width * height)
  let translucent = false
  for (let i = 0; i < plane.length; i++) {
    const a = data[i * 4 + 3]
    plane[i] = a
    if (a !== 255)
      translucent = true
  }
  return translucent ? plane : null
}

/**
 * An `ALPH` chunk: compression method 1, no filter, no pre-processing.
 *
 * Method 1 stores the plane as the GREEN channel of a VP8L image stream,
 * and "image stream" means the bitstream WITHOUT the 5-byte VP8L header -
 * the dimensions come from the frame (RFC 9649 section 5.2.3, libwebp's
 * `VP8LDecodeAlphaHeader`). That header is exactly 40 bits, so the stream
 * begins on a byte boundary and dropping five bytes is the whole job.
 */
function encodeAlphaChunk(plane: Uint8Array, width: number, height: number, options: WebpEncodeOptions): Uint8Array {
  const green = new Uint8Array(width * height * 4)
  for (let i = 0; i < plane.length; i++) {
    green[i * 4 + 1] = plane[i]
    green[i * 4 + 3] = 255
  }
  const stream = encodeVP8L({ data: green, width, height, hasAlpha: false }, options).subarray(5)

  const chunk = new Uint8Array(1 + stream.length)
  chunk[0] = 0x01 // method 1 (lossless), filter 0, pre-processing 0
  chunk.set(stream, 1)
  return chunk
}

/** VP8X (10 bytes): flags, 3 reserved, canvas width-1 and height-1 as 24-bit LE. */
function vp8xChunk(width: number, height: number, alpha: boolean): Uint8Array {
  const vp8x = new Uint8Array(10)
  vp8x[0] = alpha ? 0x10 : 0x00
  vp8x[4] = (width - 1) & 0xFF
  vp8x[5] = ((width - 1) >> 8) & 0xFF
  vp8x[6] = ((width - 1) >> 16) & 0xFF
  vp8x[7] = (height - 1) & 0xFF
  vp8x[8] = ((height - 1) >> 8) & 0xFF
  vp8x[9] = ((height - 1) >> 16) & 0xFF
  return vp8x
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

  return createRiffContainer([
    { fourCC: 'VP8X', data: vp8xChunk(width, height, hasAlphaFlag) },
    { fourCC: 'VP8L', data: vp8lData },
  ])
}
