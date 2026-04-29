import type { WebpImageData } from '../types'
import { parseVP8Header } from './header'

/**
 * VP8 (lossy WebP) decoder.
 *
 * **Status:** the foundation is in place — frame header parsing,
 * boolean arithmetic decoder, segmentation/filter/quantiser metadata
 * extraction — but the pixel-reconstruction stages (mode info,
 * coefficient decoding, inverse DCT/WHT, intra prediction, loop filter,
 * YUV→RGB) are not yet implemented. A complete VP8 decoder is roughly
 * 1500-2000 lines of careful spec-following code; building it without a
 * reference test corpus to validate against would risk shipping a
 * buggy decoder that returns wrong-but-plausible pixels.
 *
 * Today, calling `decodeVP8` parses the header (so any obvious
 * malformed input fails fast with a precise error) and then throws
 * with a clear "pixel decode not yet implemented" message. The
 * structure is right, so contributors can fill in coefficient decoding
 * and inverse transforms incrementally without restructuring the
 * dispatch.
 *
 * If you need lossy WebP decoding now, use `sharp` / `sharp-wasm` /
 * a browser's `Image()`. For pure-TS decoding, re-encode source files
 * as lossless (`cwebp -lossless`) — VP8L round-trips fully through
 * this library.
 */
export function decodeVP8(data: Uint8Array): WebpImageData {
  // Parse the frame header. This validates the start code + dimensions
  // up-front, so a caller that hands us garbage gets a useful error
  // rather than a generic "pixel decode failed" later. The parsed
  // structure also includes everything a future pixel-decoder needs.
  const header = parseVP8Header(data)

  throw new Error(
    `ts-webp: VP8 (lossy) pixel decode is not yet implemented. `
    + `The frame header parsed successfully (${header.frame.width}×${header.frame.height}, `
    + `${header.numPartitions} partition${header.numPartitions === 1 ? '' : 's'}, `
    + `quantiser ${header.quantiser.yacQi}), but coefficient decoding, `
    + `inverse DCT/WHT, intra prediction, loop filter, and YUV→RGB `
    + `conversion still need to be wired up. Re-encode as VP8L lossless `
    + `for a pure-TS round-trip, or use sharp / sharp-wasm for lossy.`,
  )
}
