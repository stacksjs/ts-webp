/**
 * VP8 intra prediction.
 *
 * Three families of predictors:
 *   - 16×16 luma: DC/V/H/TM (4 modes) — bulk-fills a 16×16 patch from
 *     the row above and column to the left.
 *   - 4×4 luma (B-modes, 10 of them): finer-grained directional
 *     predictors used when the macroblock is in `B_PRED` mode.
 *   - 8×8 chroma: same 4 modes as 16×16 luma but applied to the U
 *     and V planes.
 *
 * All predictors take a destination buffer plus references to the row
 * directly above (`top`) and the column directly to the left (`left`).
 * Edge macroblocks/sub-blocks use the spec's special boundary values:
 *   - top row of the frame:    top[i] = 127
 *   - left column of the frame: left[i] = 129
 *   - top-left corner:          129
 *
 * Reference: RFC 6386 §12, libvpx `vp8/common/reconintra4.c`.
 */

import {
  B_DC_PRED,
  B_HD_PRED,
  B_HE_PRED,
  B_HU_PRED,
  B_LD_PRED,
  B_RD_PRED,
  B_TM_PRED,
  B_VE_PRED,
  B_VL_PRED,
  B_VR_PRED,
  DC_PRED,
  H_PRED,
  TM_PRED,
  V_PRED,
} from './tables'
import { clip255 } from './idct'

// ---------------------------------------------------------------------------
// 16×16 luma intra prediction (also used for 8×8 chroma with n=8)
// ---------------------------------------------------------------------------

export function predict16x16(
  mode: number,
  dst: Uint8Array,
  stride: number,
  off: number,
  top: Uint8Array,
  left: Uint8Array,
  topLeft: number,
  hasTop: boolean,
  hasLeft: boolean,
): void {
  predictNxN(mode, dst, stride, off, top, left, topLeft, hasTop, hasLeft, 16)
}

export function predictChroma8x8(
  mode: number,
  dst: Uint8Array,
  stride: number,
  off: number,
  top: Uint8Array,
  left: Uint8Array,
  topLeft: number,
  hasTop: boolean,
  hasLeft: boolean,
): void {
  predictNxN(mode, dst, stride, off, top, left, topLeft, hasTop, hasLeft, 8)
}

function predictNxN(
  mode: number,
  dst: Uint8Array, stride: number, off: number,
  top: Uint8Array, left: Uint8Array, topLeft: number,
  hasTop: boolean, hasLeft: boolean,
  n: number,
): void {
  switch (mode) {
    case V_PRED:
      for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++)
          dst[off + y * stride + x] = top[x]
      break
    case H_PRED:
      for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++)
          dst[off + y * stride + x] = left[y]
      break
    case TM_PRED:
      for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++)
          dst[off + y * stride + x] = clip255(top[x] + left[y] - topLeft)
      break
    case DC_PRED:
    default: {
      let sum = 0
      let count = 0
      if (hasTop) {
        for (let i = 0; i < n; i++) sum += top[i]
        count += n
      }
      if (hasLeft) {
        for (let i = 0; i < n; i++) sum += left[i]
        count += n
      }
      const dc = count === 0 ? 128 : (sum + (count >> 1)) / count | 0
      for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++)
          dst[off + y * stride + x] = dc
      break
    }
  }
}

// ---------------------------------------------------------------------------
// 4×4 luma intra prediction (B-modes)
// ---------------------------------------------------------------------------

/**
 * Predict a 4×4 sub-block. Inputs:
 *   - `dst` / `stride` / `off`: destination 4×4 patch
 *   - `top`: 8 pixels above the block — top[0..3] = direct above (A,B,C,D),
 *     top[4..7] = above-right extension (E,F,G,H). Missing pixels use 127.
 *   - `left`: 4 pixels to the left of the block (rows y..y+3) (I,J,K,L).
 *   - `topLeft` (X): the single pixel at (above, left) corner.
 *
 * Implementation follows libvpx's reconintra4.c exactly. Each output
 * pixel is assigned explicitly so the diagonal modes are immune to
 * indexing errors.
 */
export function predictB(
  mode: number,
  dst: Uint8Array,
  stride: number,
  off: number,
  top: Uint8Array,
  left: Uint8Array,
  topLeft: number,
): void {
  const A = top[0], B = top[1], C = top[2], D = top[3]
  const E = top[4], F = top[5], G = top[6], H = top[7]
  const I = left[0], J = left[1], K = left[2], L = left[3]
  const X = topLeft

  // Cell-write helper.
  const w = (x: number, y: number, v: number): void => {
    dst[off + y * stride + x] = clip255(v)
  }

  // Average filters.
  const a2 = (a: number, b: number): number => (a + b + 1) >> 1
  const a3 = (a: number, b: number, c: number): number => (a + 2 * b + c + 2) >> 2

  switch (mode) {
    case B_DC_PRED: {
      const dc = (A + B + C + D + I + J + K + L + 4) >> 3
      for (let y = 0; y < 4; y++)
        for (let x = 0; x < 4; x++)
          w(x, y, dc)
      break
    }
    case B_TM_PRED: {
      for (let y = 0; y < 4; y++) {
        const ly = left[y]
        for (let x = 0; x < 4; x++)
          w(x, y, top[x] + ly - X)
      }
      break
    }
    case B_VE_PRED: {
      const v0 = a3(X, A, B)
      const v1 = a3(A, B, C)
      const v2 = a3(B, C, D)
      const v3 = a3(C, D, E)
      for (let y = 0; y < 4; y++) {
        w(0, y, v0); w(1, y, v1); w(2, y, v2); w(3, y, v3)
      }
      break
    }
    case B_HE_PRED: {
      const h0 = a3(X, I, J)
      const h1 = a3(I, J, K)
      const h2 = a3(J, K, L)
      const h3 = a3(K, L, L)
      for (let x = 0; x < 4; x++) {
        w(x, 0, h0); w(x, 1, h1); w(x, 2, h2); w(x, 3, h3)
      }
      break
    }
    case B_LD_PRED: {
      // 45° down-left. Uses A..H.
      const d0 = a3(A, B, C)
      const d1 = a3(B, C, D)
      const d2 = a3(C, D, E)
      const d3 = a3(D, E, F)
      const d4 = a3(E, F, G)
      const d5 = a3(F, G, H)
      const d6 = a3(G, H, H)
      w(0, 0, d0)
      w(1, 0, d1); w(0, 1, d1)
      w(2, 0, d2); w(1, 1, d2); w(0, 2, d2)
      w(3, 0, d3); w(2, 1, d3); w(1, 2, d3); w(0, 3, d3)
      w(3, 1, d4); w(2, 2, d4); w(1, 3, d4)
      w(3, 2, d5); w(2, 3, d5)
      w(3, 3, d6)
      break
    }
    case B_RD_PRED: {
      // 45° down-right. Diagonal indexed by (col - row + 3).
      const e0 = a3(L, K, J)
      const e1 = a3(K, J, I)
      const e2 = a3(J, I, X)
      const e3 = a3(I, X, A)
      const e4 = a3(X, A, B)
      const e5 = a3(A, B, C)
      const e6 = a3(B, C, D)
      // (col, row) → e[col - row + 3]
      w(0, 3, e0)
      w(0, 2, e1); w(1, 3, e1)
      w(0, 1, e2); w(1, 2, e2); w(2, 3, e2)
      w(0, 0, e3); w(1, 1, e3); w(2, 2, e3); w(3, 3, e3)
      w(1, 0, e4); w(2, 1, e4); w(3, 2, e4)
      w(2, 0, e5); w(3, 1, e5)
      w(3, 0, e6)
      break
    }
    case B_VR_PRED: {
      w(0, 0, a2(X, A))
      w(1, 0, a2(A, B))
      w(2, 0, a2(B, C))
      w(3, 0, a2(C, D))

      w(0, 1, a3(I, X, A))
      w(1, 1, a3(X, A, B))
      w(2, 1, a3(A, B, C))
      w(3, 1, a3(B, C, D))

      w(0, 2, a2(X, A))
      w(1, 2, a2(A, B))
      w(2, 2, a2(B, C))
      w(3, 2, a2(C, D))

      w(0, 3, a3(J, I, X))
      w(1, 3, a3(I, X, A))
      w(2, 3, a3(X, A, B))
      w(3, 3, a3(A, B, C))
      break
    }
    case B_VL_PRED: {
      w(0, 0, a2(A, B))
      w(1, 0, a2(B, C))
      w(2, 0, a2(C, D))
      w(3, 0, a2(D, E))

      w(0, 1, a3(A, B, C))
      w(1, 1, a3(B, C, D))
      w(2, 1, a3(C, D, E))
      w(3, 1, a3(D, E, F))

      w(0, 2, a2(B, C))
      w(1, 2, a2(C, D))
      w(2, 2, a2(D, E))
      w(3, 2, a2(E, F))

      w(0, 3, a3(B, C, D))
      w(1, 3, a3(C, D, E))
      w(2, 3, a3(D, E, F))
      w(3, 3, a3(E, F, G))
      break
    }
    case B_HD_PRED: {
      // Horizontal-down. Diagonal indexed by (2*row - col + 3).
      const h0 = a2(L, K)
      const h1 = a3(L, K, J)
      const h2 = a2(K, J)
      const h3 = a3(K, J, I)
      const h4 = a2(J, I)
      const h5 = a3(J, I, X)
      const h6 = a2(I, X)
      const h7 = a3(I, X, A)
      const h8 = a3(X, A, B)
      const h9 = a3(A, B, C)
      // (col, row):
      w(0, 3, h0)
      w(1, 3, h1)
      w(0, 2, h2); w(2, 3, h2)
      w(1, 2, h3); w(3, 3, h3)
      w(0, 1, h4); w(2, 2, h4)
      w(1, 1, h5); w(3, 2, h5)
      w(0, 0, h6); w(2, 1, h6)
      w(1, 0, h7); w(3, 1, h7)
      w(2, 0, h8)
      w(3, 0, h9)
      break
    }
    case B_HU_PRED: {
      // Horizontal-up.
      const u0 = a2(I, J)
      const u1 = a3(I, J, K)
      const u2 = a2(J, K)
      const u3 = a3(J, K, L)
      const u4 = a2(K, L)
      const u5 = a3(K, L, L)
      // For rows ≥ 3, all pixels are L (filled below).
      w(0, 0, u0)
      w(1, 0, u1)
      w(2, 0, u2); w(0, 1, u2)
      w(3, 0, u3); w(1, 1, u3)
      w(2, 1, u4); w(0, 2, u4)
      w(3, 1, u5); w(1, 2, u5)
      w(2, 2, L); w(0, 3, L)
      w(3, 2, L); w(1, 3, L)
      w(2, 3, L); w(3, 3, L)
      break
    }
  }
}
