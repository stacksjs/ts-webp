/**
 * VP8 encode-side quantiser. Mirrors libwebp's `QuantizeBlock_C` (`enc.c`)
 * but with a simplified `VP8Matrix` — for our minimal encoder we don't
 * vary `bias`/`sharpen`/`zthresh` per-coefficient, so a single (DC, AC)
 * pair fully describes the quantisation of a 4×4 block.
 *
 * Pipeline:
 *   1. Forward DCT produces 16 raster-order int16 coefficients.
 *   2. `quantizeBlock` divides each by Q (with rounding bias) to get a
 *      `level` value, clamped to `MAX_LEVEL = 2047` per VP8 spec.
 *   3. The encoder serialises `level` via the token tree into the
 *      boolean coder.
 *   4. To keep the prediction source consistent with what the decoder
 *      will see, we *re-multiply* each level by Q and write the
 *      reconstructed coefficient back into the input buffer — the
 *      decoder will produce these values via its own dequant pass.
 *
 * The bias term mirrors libwebp's default rounding (`bias = 32768 / 2 = 16384`
 * scaled by `iQ`). We use a fixed mid-tread offset of 0.5 (rounding to
 * nearest) for simplicity — fine for the simple, single-quality encoder
 * we're targeting.
 *
 * Reference: RFC 6386 §11.4, libwebp `dsp/enc.c::QuantizeBlock_C`.
 */

import { ZIGZAG } from './tables'

const MAX_LEVEL = 2047

/**
 * Quantise a 4×4 block's 16 raster-order coefficients using the same Q
 * lookup the decoder uses.
 *
 * @param raster — input coefficients in raster order; reconstructed
 *                 (level × Q) values are written back here so callers
 *                 can run the inverse transform from a buffer that
 *                 matches the decoder's expected residual.
 * @param zigzag — output 16 levels in zig-zag scan order (the order the
 *                 token-tree encoder will serialise).
 * @param dqDc, dqAc — Q values for DC (index 0) / AC (other 15)
 *                     coefficients.
 *
 * @returns true if any non-zero level was produced (the "non-zero" flag
 * the encoder uses to drive the EOB-only short-circuit and the
 * mb_no_skip_coef bit).
 */
export function quantizeBlock(
  raster: Int16Array,
  rasterOff: number,
  zigzag: Int16Array,
  zigzagOff: number,
  dqDc: number,
  dqAc: number,
): boolean {
  let nonZero = false
  for (let n = 0; n < 16; n++) {
    const j = ZIGZAG[n]
    const Q = n === 0 ? dqDc : dqAc
    const c = raster[rasterOff + j]
    const sign = c < 0
    const mag = sign ? -c : c
    // Round-to-nearest, mid-tread.
    let level = ((mag + (Q >> 1)) / Q) | 0
    if (level > MAX_LEVEL) level = MAX_LEVEL
    const signedLevel = sign ? -level : level
    zigzag[zigzagOff + n] = signedLevel
    // Reconstruct the dequantised value back into the raster buffer
    // so any "what the decoder will see" lookups (e.g. for predictor
    // chains down the line) read the same numbers the decoder would.
    raster[rasterOff + j] = signedLevel * Q
    if (level !== 0) nonZero = true
  }
  return nonZero
}

/**
 * Quantise the Y2 (DC-of-each-Y-block) block. Same algorithm but the
 * first coefficient (DC of the Y2 block itself) is excluded from EOB
 * detection in libwebp's MB encoder — the encoder still serialises it
 * as a normal coefficient. We expose the same nonzero flag so callers
 * can pin the `mb_no_skip_coef` bit correctly.
 */
export function quantizeY2Block(
  raster: Int16Array,
  rasterOff: number,
  zigzag: Int16Array,
  zigzagOff: number,
  dqDc: number,
  dqAc: number,
): boolean {
  return quantizeBlock(raster, rasterOff, zigzag, zigzagOff, dqDc, dqAc)
}
