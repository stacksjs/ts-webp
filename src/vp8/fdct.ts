/**
 * VP8 forward transforms — encode-side counterparts to `idct.ts`.
 *
 * 1:1 ports of libwebp's `FTransform_C` (4×4 forward DCT) and
 * `FTransformWHT_C` (Y2 Walsh-Hadamard) from `src/dsp/enc.c`. The
 * coefficient layout on output matches `idct.ts`'s expectations: 16
 * raster-order int16 entries per 4×4 block.
 *
 * The DCT operates on the *residual* (`src - ref`), where `src` is the
 * original 4×4 patch and `ref` is the 4×4 prediction. The patch is read
 * with stride `BPS` (32) bytes between rows — same convention as the
 * decoder's `intra.ts` uses.
 *
 * Reference: RFC 6386 §14.4 (forward DCT) + §14.3 (Walsh-Hadamard).
 */

const BPS = 32

/**
 * Forward 4×4 DCT on residual `src - ref`. `src` and `ref` are byte
 * patches addressed at offsets stepping in 32-byte rows; `srcOff` /
 * `refOff` are the starting offsets. `out[outOff..outOff+16)` receives
 * the 16 quantisable coefficients in raster order (DC at index 0).
 */
export function fdct4x4(
  src: Uint8Array,
  srcOff: number,
  ref: Uint8Array,
  refOff: number,
  out: Int16Array,
  outOff: number,
): void {
  const tmp = new Int32Array(16)
  for (let i = 0; i < 4; i++) {
    const sOff = srcOff + i * BPS
    const rOff = refOff + i * BPS
    const d0 = src[sOff + 0] - ref[rOff + 0]
    const d1 = src[sOff + 1] - ref[rOff + 1]
    const d2 = src[sOff + 2] - ref[rOff + 2]
    const d3 = src[sOff + 3] - ref[rOff + 3]
    const a0 = d0 + d3
    const a1 = d1 + d2
    const a2 = d1 - d2
    const a3 = d0 - d3
    tmp[0 + i * 4] = (a0 + a1) * 8
    tmp[1 + i * 4] = (a2 * 2217 + a3 * 5352 + 1812) >> 9
    tmp[2 + i * 4] = (a0 - a1) * 8
    tmp[3 + i * 4] = (a3 * 2217 - a2 * 5352 + 937) >> 9
  }
  for (let i = 0; i < 4; i++) {
    const a0 = tmp[0 + i] + tmp[12 + i]
    const a1 = tmp[4 + i] + tmp[8 + i]
    const a2 = tmp[4 + i] - tmp[8 + i]
    const a3 = tmp[0 + i] - tmp[12 + i]
    out[outOff + 0 + i] = (a0 + a1 + 7) >> 4
    out[outOff + 4 + i] = ((a2 * 2217 + a3 * 5352 + 12000) >> 16) + (a3 !== 0 ? 1 : 0)
    out[outOff + 8 + i] = (a0 - a1 + 7) >> 4
    out[outOff + 12 + i] = (a3 * 2217 - a2 * 5352 + 51000) >> 16
  }
}

/**
 * Y2 (DC-of-each-Y-block) Walsh-Hadamard forward transform. Input is
 * 16 int16 values laid out as the DC coefficients of the 16 luma 4×4
 * blocks of a macroblock; output is 16 transformed coefficients ready
 * for the Y2 block's own quantisation pass. Mirrors libwebp's
 * `FTransformWHT_C`.
 *
 * `inOff` indexes into a Y-block coefficient buffer where every 16th
 * entry is the DC of one 4×4 block — caller arranges this so the
 * function can step by 16 to walk the four DC values per row.
 */
export function fwht4x4(
  inBuf: Int16Array,
  inOff: number,
  out: Int16Array,
  outOff: number,
): void {
  const tmp = new Int32Array(16)
  // Vertical (column) pass — `inBuf` holds the 16 DCs in raster order.
  for (let i = 0; i < 4; i++) {
    const a0 = inBuf[inOff + 0 + i * 4] + inBuf[inOff + 2 + i * 4]
    const a1 = inBuf[inOff + 1 + i * 4] + inBuf[inOff + 3 + i * 4]
    const a2 = inBuf[inOff + 1 + i * 4] - inBuf[inOff + 3 + i * 4]
    const a3 = inBuf[inOff + 0 + i * 4] - inBuf[inOff + 2 + i * 4]
    tmp[0 + i * 4] = a0 + a1
    tmp[1 + i * 4] = a3 + a2
    tmp[2 + i * 4] = a3 - a2
    tmp[3 + i * 4] = a0 - a1
  }
  // Horizontal (row) pass — output the 16 transformed DCs.
  for (let i = 0; i < 4; i++) {
    const a0 = tmp[0 + i] + tmp[8 + i]
    const a1 = tmp[4 + i] + tmp[12 + i]
    const a2 = tmp[4 + i] - tmp[12 + i]
    const a3 = tmp[0 + i] - tmp[8 + i]
    out[outOff + 0 + i] = (a0 + a1) >> 1
    out[outOff + 4 + i] = (a3 + a2) >> 1
    out[outOff + 8 + i] = (a3 - a2) >> 1
    out[outOff + 12 + i] = (a0 - a1) >> 1
  }
}
