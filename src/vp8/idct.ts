/**
 * VP8 inverse transforms.
 *
 * Two transforms are used:
 *   - 4×4 IDCT — inverse of a 4-point integer DCT, applied to each
 *     non-Y2 block. Adds the residual onto the predicted pixels.
 *   - 4×4 WHT (Walsh-Hadamard) — applied to the 16 luma DCs
 *     stored in the Y2 block when 16×16 prediction is used.
 *
 * Both implementations are bit-exact with libvpx's `vp8_short_idct4x4`
 * and `vp8_short_inv_walsh4x4`. Constants are the integer fixed-point
 * approximations from RFC 6386 §14.4 / §14.3.
 */

const COSPI8SQRT2MINUS1 = 20091 // floor((cos(pi/8) * sqrt(2) - 1) << 16) lo16
const SINPI8SQRT2 = 35468 // floor( sin(pi/8) * sqrt(2)        << 16) lo16

/**
 * 4×4 IDCT. `coeffs` is 16 quantised-then-dequantised coefficients in
 * raster order (so caller must un-zigzag first if they're zigzag-ordered).
 * `dst` is a 4×4 patch (16 entries, raster order) that the residual
 * is added to and clamped to [0,255]. The patch is read+written so the
 * caller can pre-fill it with the prediction.
 */
export function idct4x4Add(coeffs: Int16Array, dst: Uint8Array, dstStride: number, dstOffset: number): void {
  // Stage 1 — process each column.
  const tmp = new Int32Array(16)
  for (let i = 0; i < 4; i++) {
    const a1 = coeffs[i] + coeffs[8 + i]
    const b1 = coeffs[i] - coeffs[8 + i]
    const c1 = ((coeffs[4 + i] * SINPI8SQRT2) >> 16) - (coeffs[12 + i] + ((coeffs[12 + i] * COSPI8SQRT2MINUS1) >> 16))
    const d1 = (coeffs[4 + i] + ((coeffs[4 + i] * COSPI8SQRT2MINUS1) >> 16)) + ((coeffs[12 + i] * SINPI8SQRT2) >> 16)
    tmp[i] = a1 + d1
    tmp[12 + i] = a1 - d1
    tmp[4 + i] = b1 + c1
    tmp[8 + i] = b1 - c1
  }

  // Stage 2 — process each row, then add+clamp into dst.
  for (let i = 0; i < 4; i++) {
    const r = i * 4
    const a1 = tmp[r + 0] + tmp[r + 2]
    const b1 = tmp[r + 0] - tmp[r + 2]
    const c1 = ((tmp[r + 1] * SINPI8SQRT2) >> 16) - (tmp[r + 3] + ((tmp[r + 3] * COSPI8SQRT2MINUS1) >> 16))
    const d1 = (tmp[r + 1] + ((tmp[r + 1] * COSPI8SQRT2MINUS1) >> 16)) + ((tmp[r + 3] * SINPI8SQRT2) >> 16)
    const o0 = (a1 + d1 + 4) >> 3
    const o3 = (a1 - d1 + 4) >> 3
    const o1 = (b1 + c1 + 4) >> 3
    const o2 = (b1 - c1 + 4) >> 3
    const base = dstOffset + i * dstStride
    dst[base + 0] = clip255(dst[base + 0] + o0)
    dst[base + 1] = clip255(dst[base + 1] + o1)
    dst[base + 2] = clip255(dst[base + 2] + o2)
    dst[base + 3] = clip255(dst[base + 3] + o3)
  }
}

/**
 * 4×4 inverse Walsh-Hadamard for the Y2 block. Decoded coefficients
 * (16 entries, raster order) are transformed in-place; the result is
 * the 16 DCs that the per-block IDCT will inject as `coeffs[0]` of
 * each of the 16 luma sub-blocks.
 */
export function iwht4x4(io: Int16Array): void {
  const tmp = new Int32Array(16)
  for (let i = 0; i < 4; i++) {
    const a1 = io[i + 0] + io[i + 12]
    const b1 = io[i + 4] + io[i + 8]
    const c1 = io[i + 4] - io[i + 8]
    const d1 = io[i + 0] - io[i + 12]
    tmp[i + 0] = a1 + b1
    tmp[i + 4] = c1 + d1
    tmp[i + 8] = a1 - b1
    tmp[i + 12] = d1 - c1
  }
  for (let i = 0; i < 4; i++) {
    const off = i * 4
    const a2 = tmp[off + 0] + tmp[off + 3]
    const b2 = tmp[off + 1] + tmp[off + 2]
    const c2 = tmp[off + 1] - tmp[off + 2]
    const d2 = tmp[off + 0] - tmp[off + 3]
    io[off + 0] = (a2 + b2 + 3) >> 3
    io[off + 1] = (c2 + d2 + 3) >> 3
    io[off + 2] = (a2 - b2 + 3) >> 3
    io[off + 3] = (d2 - c2 + 3) >> 3
  }
}

/** Clip a value to [0, 255]. */
export function clip255(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v
}
