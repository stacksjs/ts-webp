/**
 * VP8 inverse transforms — direct ports of libwebp's `TransformOne_C`
 * (4×4 IDCT) and `TransformWHT_C` (Y2 Walsh-Hadamard) from
 * `src/dsp/dec.c`.
 *
 * The IDCT here differs slightly from RFC 6386's pseudocode in
 * libwebp's favour: the `+4` rounding is added to `dc = tmp[0] + 4`
 * during the second (horizontal) pass, and the result is `>> 3` before
 * being added onto the destination buffer. We add+clip-to-[0..255] in
 * place so callers can pre-fill `dst` with the prediction.
 */

const KC1 = 20091 + (1 << 16) // floor(cos(pi/8) * sqrt(2) * (1 << 16) + 0.5) — note +1<<16
const KC2 = 35468 // floor(sin(pi/8) * sqrt(2) * (1 << 16) + 0.5)

function mulC1(v: number): number {
  // libwebp's WEBP_TRANSFORM_AC3_MUL1: ((v) * KC1) >> 16
  return Math.imul(v, KC1) >> 16
}
function mulC2(v: number): number {
  // libwebp's WEBP_TRANSFORM_AC3_MUL2: ((v) * KC2) >> 16
  return Math.imul(v, KC2) >> 16
}

function clip8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v
}

/**
 * 4×4 IDCT. `coeffs` is 16 quantised-then-dequantised DCT coefficients
 * in raster order. `dst` is a 4×4 patch (16 entries with stride
 * `dstStride`, starting at `dstOffset`) that the residual is added to
 * and clipped to [0, 255]. The patch is read+written so the caller can
 * pre-fill it with the prediction.
 */
export function idct4x4Add(
  coeffs: Int16Array,
  dst: Uint8Array,
  dstStride: number,
  dstOffset: number,
): void {
  const C = new Int32Array(16)
  // Vertical (column) pass.
  for (let i = 0; i < 4; i++) {
    const a = coeffs[i + 0] + coeffs[i + 8]
    const b = coeffs[i + 0] - coeffs[i + 8]
    const c = mulC2(coeffs[i + 4]) - mulC1(coeffs[i + 12])
    const d = mulC1(coeffs[i + 4]) + mulC2(coeffs[i + 12])
    C[i * 4 + 0] = a + d
    C[i * 4 + 1] = b + c
    C[i * 4 + 2] = b - c
    C[i * 4 + 3] = a - d
  }
  // Horizontal (row) pass with `+4` rounder folded into DC.
  for (let i = 0; i < 4; i++) {
    const dc = C[i] + 4
    const a = dc + C[i + 8]
    const b = dc - C[i + 8]
    const c = mulC2(C[i + 4]) - mulC1(C[i + 12])
    const d = mulC1(C[i + 4]) + mulC2(C[i + 12])
    const base = dstOffset + i * dstStride
    dst[base + 0] = clip8(dst[base + 0] + ((a + d) >> 3))
    dst[base + 1] = clip8(dst[base + 1] + ((b + c) >> 3))
    dst[base + 2] = clip8(dst[base + 2] + ((b - c) >> 3))
    dst[base + 3] = clip8(dst[base + 3] + ((a - d) >> 3))
  }
}

/**
 * Walsh-Hadamard transform for the Y2 block — port of libwebp's
 * `TransformWHT_C`. Decoded coefficients (16 entries, raster order)
 * are transformed in-place; the result holds the 16 DCs that the
 * per-block IDCT will inject as `coeffs[0]` of each of the 16 luma
 * sub-blocks.
 */
export function iwht4x4(io: Int16Array): void {
  const tmp = new Int32Array(16)
  for (let i = 0; i < 4; i++) {
    const a0 = io[i + 0] + io[i + 12]
    const a1 = io[i + 4] + io[i + 8]
    const a2 = io[i + 4] - io[i + 8]
    const a3 = io[i + 0] - io[i + 12]
    tmp[i + 0] = a0 + a1
    tmp[i + 8] = a0 - a1
    tmp[i + 4] = a3 + a2
    tmp[i + 12] = a3 - a2
  }
  for (let i = 0; i < 4; i++) {
    const dc = tmp[0 + i * 4] + 3
    const a0 = dc + tmp[3 + i * 4]
    const a1 = tmp[1 + i * 4] + tmp[2 + i * 4]
    const a2 = tmp[1 + i * 4] - tmp[2 + i * 4]
    const a3 = dc - tmp[3 + i * 4]
    io[0 + i * 4] = (a0 + a1) >> 3
    io[1 + i * 4] = (a3 + a2) >> 3
    io[2 + i * 4] = (a0 - a1) >> 3
    io[3 + i * 4] = (a3 - a2) >> 3
  }
}

/**
 * libwebp also has `TransformDC` and `TransformDCUV` shortcuts that
 * skip the full IDCT when only the DC coefficient is non-zero — they
 * compute `dc = (coef[0] + 4) >> 3` and add `dc` to every pixel.
 * We provide it here for use by the orchestrator.
 */
export function transformDC(coef: Int16Array, dst: Uint8Array, dstStride: number, dstOffset: number): void {
  const dc = (coef[0] + 4) >> 3
  for (let r = 0; r < 4; r++) {
    const base = dstOffset + r * dstStride
    dst[base + 0] = clip8(dst[base + 0] + dc)
    dst[base + 1] = clip8(dst[base + 1] + dc)
    dst[base + 2] = clip8(dst[base + 2] + dc)
    dst[base + 3] = clip8(dst[base + 3] + dc)
  }
}

/** Clip a value to [0, 255]. */
export function clip255(v: number): number {
  return clip8(v)
}
