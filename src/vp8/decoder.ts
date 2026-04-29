import type { WebpImageData } from '../types'

/**
 * VP8 (lossy WebP) decoder — *not implemented*.
 *
 * VP8 is genuinely large: a full decoder needs DCT/WHT coefficient
 * decoding, intra prediction across 4×4 and 16×16 modes, the boolean
 * (arithmetic) decoder, dequantisation, loop-filter post-processing, and
 * a lot of error-prone bit-twiddling. The previous version of this file
 * pretended to decode VP8 by parsing the frame header and then returning
 * an all-gray pixel buffer — strictly worse than throwing, since
 * downstream code couldn't tell the difference between "decoded a real
 * gray image" and "decoder gave up".
 *
 * If you reach here, the file is a lossy WebP (`VP8 ` chunk). Re-encode
 * as lossless (`VP8L`) with libwebp / cwebp / a browser if you need a
 * pure-TS round-trip, or use Sharp / sharp-wasm for lossy.
 */
export function decodeVP8(_data: Uint8Array): WebpImageData {
  throw new Error(
    'ts-webp: VP8 (lossy) decoding is not implemented. '
    + 'This image uses a `VP8 ` chunk (lossy WebP); only `VP8L` (lossless) '
    + 'is currently supported. Re-encode as lossless or use a different decoder.',
  )
}
