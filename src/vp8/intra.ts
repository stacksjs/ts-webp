/**
 * VP8 intra prediction.
 *
 * 1:1 port of libwebp's `VP8PredLuma16`, `VP8PredLuma4`, and
 * `VP8PredChroma8` (src/dsp/dec.c). Predictors read neighbour samples
 * directly from the destination buffer at offsets `-1` (left column),
 * `-BPS` (row above), and `-1 - BPS` (top-left corner). The caller is
 * responsible for arranging the buffer with the right surrounding
 * context — see decoder.ts `ReconstructRow` for the layout.
 */

/** Stride between rows in the per-macroblock YUV buffer. */
export const BPS = 32

// Mode constants — libwebp keeps separate "no-top", "no-left",
// "no-top-left" variants of 16×16/chroma DC for the missing-neighbour
// edge cases.
export const DC_PRED = 0
export const V_PRED = 1
export const H_PRED = 2
export const TM_PRED = 3
export const B_PRED = 4
export const B_DC_PRED_NOTOP = 4
export const B_DC_PRED_NOLEFT = 5
export const B_DC_PRED_NOTOPLEFT = 6
export const NUM_B_DC_MODES = 7

// 4×4 B-modes
export const B_DC = 0
export const B_TM = 1
export const B_VE = 2
export const B_HE = 3
export const B_LD = 4
export const B_RD = 5
export const B_VR = 6
export const B_VL = 7
export const B_HD = 8
export const B_HU = 9
export const NUM_BMODES = 10

function clip8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v
}

// ---------------------------------------------------------------------------
// 16×16 luma predictors (port of libwebp's `VP8PredLuma16` table)
// ---------------------------------------------------------------------------

function ve16(dst: Uint8Array, off: number): void {
  // Vertical: copy row above into all 16 rows.
  for (let j = 0; j < 16; j++) {
    for (let x = 0; x < 16; x++) dst[off + j * BPS + x] = dst[off - BPS + x]
  }
}

function he16(dst: Uint8Array, off: number): void {
  // Horizontal: replicate left column across each row.
  for (let j = 0; j < 16; j++) {
    const v = dst[off + j * BPS - 1]
    for (let x = 0; x < 16; x++) dst[off + j * BPS + x] = v
  }
}

function put16(value: number, dst: Uint8Array, off: number): void {
  for (let j = 0; j < 16; j++)
    for (let x = 0; x < 16; x++)
      dst[off + j * BPS + x] = value
}

function dc16(dst: Uint8Array, off: number): void {
  let dc = 16
  for (let j = 0; j < 16; j++) dc += dst[off - 1 + j * BPS] + dst[off + j - BPS]
  put16(dc >> 5, dst, off)
}

function dc16NoTop(dst: Uint8Array, off: number): void {
  let dc = 8
  for (let j = 0; j < 16; j++) dc += dst[off - 1 + j * BPS]
  put16(dc >> 4, dst, off)
}

function dc16NoLeft(dst: Uint8Array, off: number): void {
  let dc = 8
  for (let i = 0; i < 16; i++) dc += dst[off + i - BPS]
  put16(dc >> 4, dst, off)
}

function dc16NoTopLeft(dst: Uint8Array, off: number): void {
  put16(0x80, dst, off)
}

function tm16(dst: Uint8Array, off: number): void {
  trueMotion(dst, off, 16)
}

/** Table-driven 16×16 dispatch. */
export const VP8PredLuma16: ReadonlyArray<(dst: Uint8Array, off: number) => void> = [
  dc16, // DC_PRED
  tm16, // TM_PRED — note libwebp orders modes differently; we apply mapping below
  ve16, // V_PRED
  he16, // H_PRED
  dc16NoTop, // B_DC_PRED_NOTOP
  dc16NoLeft, // B_DC_PRED_NOLEFT
  dc16NoTopLeft, // B_DC_PRED_NOTOPLEFT
]

/**
 * Apply a 16×16 luma predictor by mode. The `mode` value is the
 * decoder's logical mode (DC_PRED=0, V_PRED=1, H_PRED=2, TM_PRED=3) or
 * one of the variants used at frame edges.
 */
export function predictLuma16(mode: number, dst: Uint8Array, off: number): void {
  switch (mode) {
    case DC_PRED: dc16(dst, off); break
    case V_PRED: ve16(dst, off); break
    case H_PRED: he16(dst, off); break
    case TM_PRED: tm16(dst, off); break
    case B_DC_PRED_NOTOP: dc16NoTop(dst, off); break
    case B_DC_PRED_NOLEFT: dc16NoLeft(dst, off); break
    case B_DC_PRED_NOTOPLEFT: dc16NoTopLeft(dst, off); break
  }
}

// ---------------------------------------------------------------------------
// 8×8 chroma predictors
// ---------------------------------------------------------------------------

function put8x8(value: number, dst: Uint8Array, off: number): void {
  for (let j = 0; j < 8; j++)
    for (let x = 0; x < 8; x++)
      dst[off + j * BPS + x] = value
}

function dc8uv(dst: Uint8Array, off: number): void {
  let dc = 8
  for (let i = 0; i < 8; i++) dc += dst[off + i - BPS] + dst[off - 1 + i * BPS]
  put8x8(dc >> 4, dst, off)
}

function dc8uvNoTop(dst: Uint8Array, off: number): void {
  let dc = 4
  for (let i = 0; i < 8; i++) dc += dst[off - 1 + i * BPS]
  put8x8(dc >> 3, dst, off)
}

function dc8uvNoLeft(dst: Uint8Array, off: number): void {
  let dc = 4
  for (let i = 0; i < 8; i++) dc += dst[off + i - BPS]
  put8x8(dc >> 3, dst, off)
}

function dc8uvNoTopLeft(dst: Uint8Array, off: number): void {
  put8x8(0x80, dst, off)
}

function ve8uv(dst: Uint8Array, off: number): void {
  for (let j = 0; j < 8; j++)
    for (let x = 0; x < 8; x++)
      dst[off + j * BPS + x] = dst[off - BPS + x]
}

function he8uv(dst: Uint8Array, off: number): void {
  for (let j = 0; j < 8; j++) {
    const v = dst[off + j * BPS - 1]
    for (let x = 0; x < 8; x++) dst[off + j * BPS + x] = v
  }
}

function tm8uv(dst: Uint8Array, off: number): void {
  trueMotion(dst, off, 8)
}

export function predictChroma8(mode: number, dst: Uint8Array, off: number): void {
  switch (mode) {
    case DC_PRED: dc8uv(dst, off); break
    case V_PRED: ve8uv(dst, off); break
    case H_PRED: he8uv(dst, off); break
    case TM_PRED: tm8uv(dst, off); break
    case B_DC_PRED_NOTOP: dc8uvNoTop(dst, off); break
    case B_DC_PRED_NOLEFT: dc8uvNoLeft(dst, off); break
    case B_DC_PRED_NOTOPLEFT: dc8uvNoTopLeft(dst, off); break
  }
}

// ---------------------------------------------------------------------------
// True-motion predictor — clip(top[x] + left[y] - top_left)
// ---------------------------------------------------------------------------

function trueMotion(dst: Uint8Array, off: number, size: number): void {
  const topLeftIdx = off - 1 - BPS
  const topLeft = dst[topLeftIdx]
  for (let y = 0; y < size; y++) {
    const lefty = dst[off + y * BPS - 1]
    for (let x = 0; x < size; x++) {
      dst[off + y * BPS + x] = clip8(dst[off - BPS + x] + lefty - topLeft)
    }
  }
}

// ---------------------------------------------------------------------------
// 4×4 B-modes (port of libwebp's VE4_C/HE4_C/DC4_C/RD4_C/LD4_C/VR4_C/VL4_C/HD4_C/HU4_C/TM4_C)
// ---------------------------------------------------------------------------

const avg2 = (a: number, b: number): number => (a + b + 1) >> 1
const avg3 = (a: number, b: number, c: number): number => (a + 2 * b + c + 2) >> 2

function tm4(dst: Uint8Array, off: number): void {
  trueMotion(dst, off, 4)
}

function ve4(dst: Uint8Array, off: number): void {
  // top samples at dst[off - BPS - 1 .. off - BPS + 3]
  const t0 = dst[off - BPS - 1]
  const t1 = dst[off - BPS + 0]
  const t2 = dst[off - BPS + 1]
  const t3 = dst[off - BPS + 2]
  const t4 = dst[off - BPS + 3]
  const t5 = dst[off - BPS + 4]
  const v0 = avg3(t0, t1, t2)
  const v1 = avg3(t1, t2, t3)
  const v2 = avg3(t2, t3, t4)
  const v3 = avg3(t3, t4, t5)
  for (let i = 0; i < 4; i++) {
    const base = off + i * BPS
    dst[base + 0] = v0
    dst[base + 1] = v1
    dst[base + 2] = v2
    dst[base + 3] = v3
  }
}

function he4(dst: Uint8Array, off: number): void {
  const A = dst[off - 1 - BPS]
  const B = dst[off - 1]
  const C = dst[off - 1 + BPS]
  const D = dst[off - 1 + 2 * BPS]
  const E = dst[off - 1 + 3 * BPS]
  const v0 = avg3(A, B, C)
  const v1 = avg3(B, C, D)
  const v2 = avg3(C, D, E)
  const v3 = avg3(D, E, E)
  for (let x = 0; x < 4; x++) dst[off + 0 * BPS + x] = v0
  for (let x = 0; x < 4; x++) dst[off + 1 * BPS + x] = v1
  for (let x = 0; x < 4; x++) dst[off + 2 * BPS + x] = v2
  for (let x = 0; x < 4; x++) dst[off + 3 * BPS + x] = v3
}

function dc4(dst: Uint8Array, off: number): void {
  let dc = 4
  for (let i = 0; i < 4; i++) dc += dst[off + i - BPS] + dst[off - 1 + i * BPS]
  dc >>= 3
  for (let i = 0; i < 4; i++)
    for (let x = 0; x < 4; x++)
      dst[off + i * BPS + x] = dc
}

// Helper for the diagonal predictors — write at (col, row).
function setPixel(dst: Uint8Array, off: number, col: number, row: number, v: number): void {
  dst[off + col + row * BPS] = v
}

function rd4(dst: Uint8Array, off: number): void {
  const I = dst[off - 1 + 0 * BPS]
  const J = dst[off - 1 + 1 * BPS]
  const K = dst[off - 1 + 2 * BPS]
  const L = dst[off - 1 + 3 * BPS]
  const X = dst[off - 1 - BPS]
  const A = dst[off + 0 - BPS]
  const B = dst[off + 1 - BPS]
  const C = dst[off + 2 - BPS]
  const D = dst[off + 3 - BPS]
  setPixel(dst, off, 0, 3, avg3(J, K, L))
  setPixel(dst, off, 0, 2, avg3(I, J, K))
  setPixel(dst, off, 1, 3, avg3(I, J, K))
  setPixel(dst, off, 0, 1, avg3(X, I, J))
  setPixel(dst, off, 1, 2, avg3(X, I, J))
  setPixel(dst, off, 2, 3, avg3(X, I, J))
  setPixel(dst, off, 0, 0, avg3(A, X, I))
  setPixel(dst, off, 1, 1, avg3(A, X, I))
  setPixel(dst, off, 2, 2, avg3(A, X, I))
  setPixel(dst, off, 3, 3, avg3(A, X, I))
  setPixel(dst, off, 1, 0, avg3(B, A, X))
  setPixel(dst, off, 2, 1, avg3(B, A, X))
  setPixel(dst, off, 3, 2, avg3(B, A, X))
  setPixel(dst, off, 2, 0, avg3(C, B, A))
  setPixel(dst, off, 3, 1, avg3(C, B, A))
  setPixel(dst, off, 3, 0, avg3(D, C, B))
}

function ld4(dst: Uint8Array, off: number): void {
  const A = dst[off + 0 - BPS]
  const B = dst[off + 1 - BPS]
  const C = dst[off + 2 - BPS]
  const D = dst[off + 3 - BPS]
  const E = dst[off + 4 - BPS]
  const F = dst[off + 5 - BPS]
  const G = dst[off + 6 - BPS]
  const H = dst[off + 7 - BPS]
  setPixel(dst, off, 0, 0, avg3(A, B, C))
  setPixel(dst, off, 1, 0, avg3(B, C, D))
  setPixel(dst, off, 0, 1, avg3(B, C, D))
  setPixel(dst, off, 2, 0, avg3(C, D, E))
  setPixel(dst, off, 1, 1, avg3(C, D, E))
  setPixel(dst, off, 0, 2, avg3(C, D, E))
  setPixel(dst, off, 3, 0, avg3(D, E, F))
  setPixel(dst, off, 2, 1, avg3(D, E, F))
  setPixel(dst, off, 1, 2, avg3(D, E, F))
  setPixel(dst, off, 0, 3, avg3(D, E, F))
  setPixel(dst, off, 3, 1, avg3(E, F, G))
  setPixel(dst, off, 2, 2, avg3(E, F, G))
  setPixel(dst, off, 1, 3, avg3(E, F, G))
  setPixel(dst, off, 3, 2, avg3(F, G, H))
  setPixel(dst, off, 2, 3, avg3(F, G, H))
  setPixel(dst, off, 3, 3, avg3(G, H, H))
}

function vr4(dst: Uint8Array, off: number): void {
  const I = dst[off - 1 + 0 * BPS]
  const J = dst[off - 1 + 1 * BPS]
  const K = dst[off - 1 + 2 * BPS]
  const X = dst[off - 1 - BPS]
  const A = dst[off + 0 - BPS]
  const B = dst[off + 1 - BPS]
  const C = dst[off + 2 - BPS]
  const D = dst[off + 3 - BPS]
  setPixel(dst, off, 0, 0, avg2(X, A))
  setPixel(dst, off, 1, 2, avg2(X, A))
  setPixel(dst, off, 1, 0, avg2(A, B))
  setPixel(dst, off, 2, 2, avg2(A, B))
  setPixel(dst, off, 2, 0, avg2(B, C))
  setPixel(dst, off, 3, 2, avg2(B, C))
  setPixel(dst, off, 3, 0, avg2(C, D))

  setPixel(dst, off, 0, 3, avg3(K, J, I))
  setPixel(dst, off, 0, 2, avg3(J, I, X))
  setPixel(dst, off, 0, 1, avg3(I, X, A))
  setPixel(dst, off, 1, 3, avg3(I, X, A))
  setPixel(dst, off, 1, 1, avg3(X, A, B))
  setPixel(dst, off, 2, 3, avg3(X, A, B))
  setPixel(dst, off, 2, 1, avg3(A, B, C))
  setPixel(dst, off, 3, 3, avg3(A, B, C))
  setPixel(dst, off, 3, 1, avg3(B, C, D))
}

function vl4(dst: Uint8Array, off: number): void {
  const A = dst[off + 0 - BPS]
  const B = dst[off + 1 - BPS]
  const C = dst[off + 2 - BPS]
  const D = dst[off + 3 - BPS]
  const E = dst[off + 4 - BPS]
  const F = dst[off + 5 - BPS]
  const G = dst[off + 6 - BPS]
  const H = dst[off + 7 - BPS]
  setPixel(dst, off, 0, 0, avg2(A, B))
  setPixel(dst, off, 1, 0, avg2(B, C))
  setPixel(dst, off, 0, 2, avg2(B, C))
  setPixel(dst, off, 2, 0, avg2(C, D))
  setPixel(dst, off, 1, 2, avg2(C, D))
  setPixel(dst, off, 3, 0, avg2(D, E))
  setPixel(dst, off, 2, 2, avg2(D, E))

  setPixel(dst, off, 0, 1, avg3(A, B, C))
  setPixel(dst, off, 1, 1, avg3(B, C, D))
  setPixel(dst, off, 0, 3, avg3(B, C, D))
  setPixel(dst, off, 2, 1, avg3(C, D, E))
  setPixel(dst, off, 1, 3, avg3(C, D, E))
  setPixel(dst, off, 3, 1, avg3(D, E, F))
  setPixel(dst, off, 2, 3, avg3(D, E, F))
  setPixel(dst, off, 3, 2, avg3(E, F, G))
  setPixel(dst, off, 3, 3, avg3(F, G, H))
}

function hu4(dst: Uint8Array, off: number): void {
  const I = dst[off - 1 + 0 * BPS]
  const J = dst[off - 1 + 1 * BPS]
  const K = dst[off - 1 + 2 * BPS]
  const L = dst[off - 1 + 3 * BPS]
  setPixel(dst, off, 0, 0, avg2(I, J))
  setPixel(dst, off, 2, 0, avg2(J, K))
  setPixel(dst, off, 0, 1, avg2(J, K))
  setPixel(dst, off, 2, 1, avg2(K, L))
  setPixel(dst, off, 0, 2, avg2(K, L))
  setPixel(dst, off, 1, 0, avg3(I, J, K))
  setPixel(dst, off, 3, 0, avg3(J, K, L))
  setPixel(dst, off, 1, 1, avg3(J, K, L))
  setPixel(dst, off, 3, 1, avg3(K, L, L))
  setPixel(dst, off, 1, 2, avg3(K, L, L))
  setPixel(dst, off, 3, 2, L)
  setPixel(dst, off, 2, 2, L)
  setPixel(dst, off, 0, 3, L)
  setPixel(dst, off, 1, 3, L)
  setPixel(dst, off, 2, 3, L)
  setPixel(dst, off, 3, 3, L)
}

function hd4(dst: Uint8Array, off: number): void {
  const I = dst[off - 1 + 0 * BPS]
  const J = dst[off - 1 + 1 * BPS]
  const K = dst[off - 1 + 2 * BPS]
  const L = dst[off - 1 + 3 * BPS]
  const X = dst[off - 1 - BPS]
  const A = dst[off + 0 - BPS]
  const B = dst[off + 1 - BPS]
  const C = dst[off + 2 - BPS]
  setPixel(dst, off, 0, 0, avg2(I, X))
  setPixel(dst, off, 2, 1, avg2(I, X))
  setPixel(dst, off, 0, 1, avg2(J, I))
  setPixel(dst, off, 2, 2, avg2(J, I))
  setPixel(dst, off, 0, 2, avg2(K, J))
  setPixel(dst, off, 2, 3, avg2(K, J))
  setPixel(dst, off, 0, 3, avg2(L, K))

  setPixel(dst, off, 3, 0, avg3(A, B, C))
  setPixel(dst, off, 2, 0, avg3(X, A, B))
  setPixel(dst, off, 1, 0, avg3(I, X, A))
  setPixel(dst, off, 3, 1, avg3(I, X, A))
  setPixel(dst, off, 1, 1, avg3(J, I, X))
  setPixel(dst, off, 3, 2, avg3(J, I, X))
  setPixel(dst, off, 1, 2, avg3(K, J, I))
  setPixel(dst, off, 3, 3, avg3(K, J, I))
  setPixel(dst, off, 1, 3, avg3(L, K, J))
}

/** Dispatch a 4×4 B-mode predictor by mode index. */
export function predictLuma4(mode: number, dst: Uint8Array, off: number): void {
  switch (mode) {
    case B_DC: dc4(dst, off); break
    case B_TM: tm4(dst, off); break
    case B_VE: ve4(dst, off); break
    case B_HE: he4(dst, off); break
    case B_LD: ld4(dst, off); break
    case B_RD: rd4(dst, off); break
    case B_VR: vr4(dst, off); break
    case B_VL: vl4(dst, off); break
    case B_HD: hd4(dst, off); break
    case B_HU: hu4(dst, off); break
  }
}
