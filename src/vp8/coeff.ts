/**
 * VP8 DCT coefficient decoder.
 *
 * 1:1 port of libwebp's `GetCoeffsFast` (src/dec/vp8_dec.c §13.2/13.3).
 * Reads 16 quantised DCT coefficients of one 4×4 block from a `BoolDecoder`,
 * writes them in raster order via `kZigzag`, and applies dequantisation
 * inline using a 2-element `[dc, ac]` matrix.
 *
 * The probability layout used here is libwebp's: a flat
 * `[block_type][band][ctx][prob_idx]` table (`DEFAULT_COEF_PROBS`),
 * indexed via the precomputed band map `COEF_BANDS`. The decoder advances
 * a `(block_type, n, ctx)` triple through the bitstream where `n` is the
 * zig-zag scan position; the next position's probabilities are picked up
 * before reading each subsequent token.
 */
import { BoolDecoder } from './bool-decoder'
import { CAT3456_PROBS, COEF_BANDS, NUM_COEF_BANDS, ZIGZAG } from './tables'

/** Block-type indices match libwebp's NUM_TYPES enum. */
export const BLOCK_TYPE_Y_AFTER_Y2 = 0
export const BLOCK_TYPE_Y2 = 1
export const BLOCK_TYPE_UV = 2
export const BLOCK_TYPE_Y_NO_Y2 = 3

/**
 * Decode the magnitude of a "large" coefficient (≥ 2). Mirrors
 * libwebp's `GetLargeValue`. `p` is the 11-element probability vector
 * for the current zig-zag position with the relevant ctx.
 */
function getLargeValue(br: BoolDecoder, p: Uint8Array): number {
  let v: number
  if (!br.readBit(p[3])) {
    if (!br.readBit(p[4])) {
      v = 2
    }
    else {
      v = 3 + br.readBit(p[5])
    }
  }
  else {
    if (!br.readBit(p[6])) {
      if (!br.readBit(p[7])) {
        v = 5 + br.readBit(159)
      }
      else {
        v = 7 + 2 * br.readBit(165)
        v += br.readBit(145)
      }
    }
    else {
      const bit1 = br.readBit(p[8])
      const bit0 = br.readBit(p[9 + bit1])
      const cat = 2 * bit1 + bit0
      const tab = CAT3456_PROBS[cat]
      v = 0
      for (let i = 0; tab[i] !== 0; i++) {
        v += v + br.readBit(tab[i])
      }
      v += 3 + (8 << cat)
    }
  }
  return v
}

/**
 * Slice helper — produce a Uint8Array view of the 11 probability bytes
 * for `(block_type, band, ctx)`.
 */
function probSlice(probs: Uint8Array, blockType: number, band: number, ctx: number): Uint8Array {
  const off = blockType * NUM_COEF_BANDS * 3 * 11 + band * 3 * 11 + ctx * 11
  return probs.subarray(off, off + 11)
}

/**
 * Decode the DCT coefficients of one 4×4 block. The output is written in
 * raster order via the zig-zag scan and dequantised with `dq = [dc, ac]`.
 *
 * `ctx` is the (above_nz + left_nz) context for the block's first
 * coefficient. `firstCoef` is 0 for blocks that include the DC term, 1
 * for luma blocks of a 16×16-predicted macroblock (DC lives in the Y2
 * block instead).
 *
 * Returns the position of the last non-zero coefficient plus one (libwebp's
 * convention) — i.e. 0 if the block was empty (immediate EOB), 16 if all
 * 16 positions were filled, and a value in (firstCoef, 16) otherwise.
 */
export function decodeBlockCoeffs(
  br: BoolDecoder,
  out: Int16Array,
  blockType: number,
  ctx: number,
  firstCoef: number,
  probs: Uint8Array,
  dqDc: number,
  dqAc: number,
): number {
  let n = firstCoef
  let p = probSlice(probs, blockType, COEF_BANDS[n], ctx)

  for (; n < 16; n++) {
    if (!br.readBit(p[0])) {
      // EOB — return position of last non-zero coeff plus one.
      return n
    }
    while (!br.readBit(p[1])) {
      // sequence of zero coefs: advance n with ctx=0 probs
      n++
      if (n === 16) return 16
      p = probSlice(probs, blockType, COEF_BANDS[n], 0)
    }
    // Non-zero coef.
    let v: number
    let nextCtx: number
    if (!br.readBit(p[2])) {
      v = 1
      nextCtx = 1
    }
    else {
      v = getLargeValue(br, p)
      nextCtx = 2
    }
    // libwebp's coef sign uses VP8GetSigned (different state-advance
    // path than VP8GetBit(0x80)). Must match for bit-exact agreement.
    const dq = n > 0 ? dqAc : dqDc
    out[ZIGZAG[n]] = br.readSigned(v) * dq
    // Pre-fetch next position's probs at the chosen context.
    if (n + 1 < 16) {
      p = probSlice(probs, blockType, COEF_BANDS[n + 1], nextCtx)
    }
  }
  return 16
}
