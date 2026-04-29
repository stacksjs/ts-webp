/**
 * VP8 in-loop deblocking filter — 1:1 port of libwebp's filtering
 * primitives (`src/dsp/dec.c` `DoFilter2_C`/`DoFilter4_C`/`DoFilter6_C`,
 * `FilterLoop24_C`/`FilterLoop26_C`, and the V/H/Simple variants).
 *
 * The filter has two flavours:
 *   - Simple (filter_type == 1): a thin 4-tap kernel applied across MB
 *     and sub-block edges.
 *   - Complex (filter_type == 2): a wider kernel that branches between
 *     `DoFilter2` (high-edge-variance) and `DoFilter4`/`DoFilter6`
 *     (smooth) based on neighbour magnitudes.
 *
 * Filtering operates in-place on the reconstructed YUV planes after
 * intra prediction + residual addition.
 */

// libwebp's clip lookups — we keep them as plain inline helpers since
// JS array-index lookups aren't faster than the underlying min/max
// arithmetic on V8/JSC.

function kclip1(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v
}
function ksclip1(v: number): number {
  return v < -128 ? -128 : v > 127 ? 127 : v
}
function ksclip2(v: number): number {
  return v < -16 ? -16 : v > 15 ? 15 : v
}
function kabs(v: number): number {
  return v < 0 ? -v : v
}

// ---------------------------------------------------------------------------
// Per-pixel filter kernels (DoFilter2/4/6 from src/dsp/dec.c)
// ---------------------------------------------------------------------------

function doFilter2(p: Uint8Array, off: number, step: number): void {
  const p1 = p[off - 2 * step]
  const p0 = p[off - step]
  const q0 = p[off]
  const q1 = p[off + step]
  const a = 3 * (q0 - p0) + ksclip1(p1 - q1)
  const a1 = ksclip2((a + 4) >> 3)
  const a2 = ksclip2((a + 3) >> 3)
  p[off - step] = kclip1(p0 + a2)
  p[off] = kclip1(q0 - a1)
}

function doFilter4(p: Uint8Array, off: number, step: number): void {
  const p1 = p[off - 2 * step]
  const p0 = p[off - step]
  const q0 = p[off]
  const q1 = p[off + step]
  const a = 3 * (q0 - p0)
  const a1 = ksclip2((a + 4) >> 3)
  const a2 = ksclip2((a + 3) >> 3)
  const a3 = (a1 + 1) >> 1
  p[off - 2 * step] = kclip1(p1 + a3)
  p[off - step] = kclip1(p0 + a2)
  p[off] = kclip1(q0 - a1)
  p[off + step] = kclip1(q1 - a3)
}

function doFilter6(p: Uint8Array, off: number, step: number): void {
  const p2 = p[off - 3 * step]
  const p1 = p[off - 2 * step]
  const p0 = p[off - step]
  const q0 = p[off]
  const q1 = p[off + step]
  const q2 = p[off + 2 * step]
  const a = ksclip1(3 * (q0 - p0) + ksclip1(p1 - q1))
  const a1 = (27 * a + 63) >> 7
  const a2 = (18 * a + 63) >> 7
  const a3 = (9 * a + 63) >> 7
  p[off - 3 * step] = kclip1(p2 + a3)
  p[off - 2 * step] = kclip1(p1 + a2)
  p[off - step] = kclip1(p0 + a1)
  p[off] = kclip1(q0 - a1)
  p[off + step] = kclip1(q1 - a2)
  p[off + 2 * step] = kclip1(q2 - a3)
}

function isHev(p: Uint8Array, off: number, step: number, thresh: number): boolean {
  const p1 = p[off - 2 * step]
  const p0 = p[off - step]
  const q0 = p[off]
  const q1 = p[off + step]
  return kabs(p1 - p0) > thresh || kabs(q1 - q0) > thresh
}

function needsFilter(p: Uint8Array, off: number, step: number, t: number): boolean {
  const p1 = p[off - 2 * step]
  const p0 = p[off - step]
  const q0 = p[off]
  const q1 = p[off + step]
  return (4 * kabs(p0 - q0) + kabs(p1 - q1)) <= t
}

function needsFilter2(p: Uint8Array, off: number, step: number, t: number, it: number): boolean {
  const p3 = p[off - 4 * step]
  const p2 = p[off - 3 * step]
  const p1 = p[off - 2 * step]
  const p0 = p[off - step]
  const q0 = p[off]
  const q1 = p[off + step]
  const q2 = p[off + 2 * step]
  const q3 = p[off + 3 * step]
  if ((4 * kabs(p0 - q0) + kabs(p1 - q1)) > t) return false
  return kabs(p3 - p2) <= it && kabs(p2 - p1) <= it && kabs(p1 - p0) <= it
    && kabs(q3 - q2) <= it && kabs(q2 - q1) <= it && kabs(q1 - q0) <= it
}

// ---------------------------------------------------------------------------
// FilterLoop24/26 — apply DoFilter2/4 or DoFilter2/6 along an edge
// ---------------------------------------------------------------------------

function filterLoop26(
  p: Uint8Array, off: number, hstride: number, vstride: number,
  size: number, thresh: number, ithresh: number, hevThresh: number,
): void {
  const thresh2 = 2 * thresh + 1
  while (size-- > 0) {
    if (needsFilter2(p, off, hstride, thresh2, ithresh)) {
      if (isHev(p, off, hstride, hevThresh)) {
        doFilter2(p, off, hstride)
      }
      else {
        doFilter6(p, off, hstride)
      }
    }
    off += vstride
  }
}

function filterLoop24(
  p: Uint8Array, off: number, hstride: number, vstride: number,
  size: number, thresh: number, ithresh: number, hevThresh: number,
): void {
  const thresh2 = 2 * thresh + 1
  while (size-- > 0) {
    if (needsFilter2(p, off, hstride, thresh2, ithresh)) {
      if (isHev(p, off, hstride, hevThresh)) {
        doFilter2(p, off, hstride)
      }
      else {
        doFilter4(p, off, hstride)
      }
    }
    off += vstride
  }
}

// ---------------------------------------------------------------------------
// Simple filter (filter_type == 1)
// ---------------------------------------------------------------------------

export function simpleVFilter16(p: Uint8Array, off: number, stride: number, thresh: number): void {
  const thresh2 = 2 * thresh + 1
  for (let i = 0; i < 16; i++) {
    if (needsFilter(p, off + i, stride, thresh2)) {
      doFilter2(p, off + i, stride)
    }
  }
}

export function simpleHFilter16(p: Uint8Array, off: number, stride: number, thresh: number): void {
  const thresh2 = 2 * thresh + 1
  for (let i = 0; i < 16; i++) {
    if (needsFilter(p, off + i * stride, 1, thresh2)) {
      doFilter2(p, off + i * stride, 1)
    }
  }
}

export function simpleVFilter16i(p: Uint8Array, off: number, stride: number, thresh: number): void {
  for (let k = 3; k > 0; k--) {
    off += 4 * stride
    simpleVFilter16(p, off, stride, thresh)
  }
}

export function simpleHFilter16i(p: Uint8Array, off: number, stride: number, thresh: number): void {
  for (let k = 3; k > 0; k--) {
    off += 4
    simpleHFilter16(p, off, stride, thresh)
  }
}

// ---------------------------------------------------------------------------
// Complex filter (filter_type == 2)
// ---------------------------------------------------------------------------

export function vFilter16(p: Uint8Array, off: number, stride: number, thresh: number, ithresh: number, hevThresh: number): void {
  filterLoop26(p, off, stride, 1, 16, thresh, ithresh, hevThresh)
}

export function hFilter16(p: Uint8Array, off: number, stride: number, thresh: number, ithresh: number, hevThresh: number): void {
  filterLoop26(p, off, 1, stride, 16, thresh, ithresh, hevThresh)
}

export function vFilter16i(p: Uint8Array, off: number, stride: number, thresh: number, ithresh: number, hevThresh: number): void {
  for (let k = 3; k > 0; k--) {
    off += 4 * stride
    filterLoop24(p, off, stride, 1, 16, thresh, ithresh, hevThresh)
  }
}

export function hFilter16i(p: Uint8Array, off: number, stride: number, thresh: number, ithresh: number, hevThresh: number): void {
  for (let k = 3; k > 0; k--) {
    off += 4
    filterLoop24(p, off, 1, stride, 16, thresh, ithresh, hevThresh)
  }
}

export function vFilter8(u: Uint8Array, uOff: number, v: Uint8Array, vOff: number, stride: number, thresh: number, ithresh: number, hevThresh: number): void {
  filterLoop26(u, uOff, stride, 1, 8, thresh, ithresh, hevThresh)
  filterLoop26(v, vOff, stride, 1, 8, thresh, ithresh, hevThresh)
}

export function hFilter8(u: Uint8Array, uOff: number, v: Uint8Array, vOff: number, stride: number, thresh: number, ithresh: number, hevThresh: number): void {
  filterLoop26(u, uOff, 1, stride, 8, thresh, ithresh, hevThresh)
  filterLoop26(v, vOff, 1, stride, 8, thresh, ithresh, hevThresh)
}

export function vFilter8i(u: Uint8Array, uOff: number, v: Uint8Array, vOff: number, stride: number, thresh: number, ithresh: number, hevThresh: number): void {
  filterLoop24(u, uOff + 4 * stride, stride, 1, 8, thresh, ithresh, hevThresh)
  filterLoop24(v, vOff + 4 * stride, stride, 1, 8, thresh, ithresh, hevThresh)
}

export function hFilter8i(u: Uint8Array, uOff: number, v: Uint8Array, vOff: number, stride: number, thresh: number, ithresh: number, hevThresh: number): void {
  filterLoop24(u, uOff + 4, 1, stride, 8, thresh, ithresh, hevThresh)
  filterLoop24(v, vOff + 4, 1, stride, 8, thresh, ithresh, hevThresh)
}

// ---------------------------------------------------------------------------
// Filter-strength precompute — port of libwebp's `PrecomputeFilterStrengths`
// ---------------------------------------------------------------------------

export interface FInfo {
  /** Inner-edge filter level (libwebp's `f_ilevel`). */
  iLevel: number
  /** Outer (MB-edge) filter limit (libwebp's `f_limit`). */
  limit: number
  /** High-edge-variance threshold (libwebp's `hev_thresh`). */
  hevThresh: number
  /** Whether to filter inner sub-block edges (libwebp's `f_inner`). */
  inner: boolean
}

/**
 * Compute filter strength for a (segment, mode) pair. Mirrors
 * `PrecomputeFilterStrengths` in libwebp's frame_dec.c §15.4.
 *
 * Inputs:
 *   - `baseLevel` is the per-segment level (post-segment-override) before
 *     mode/ref deltas.
 *   - `useLfDelta`, `refDelta0`, `modeDelta0` come from the filter header.
 *   - `sharpness` is the sharpness_level field.
 *   - `i4x4` is true when the macroblock uses B_PRED (inner edges filtered).
 */
export function precomputeFilterStrength(
  baseLevel: number,
  useLfDelta: boolean,
  refDelta0: number,
  modeDelta0: number,
  sharpness: number,
  i4x4: boolean,
): FInfo {
  let level = baseLevel
  if (useLfDelta) {
    level += refDelta0
    if (i4x4) level += modeDelta0
  }
  level = level < 0 ? 0 : level > 63 ? 63 : level
  if (level === 0) {
    return { iLevel: 0, limit: 0, hevThresh: 0, inner: i4x4 }
  }
  let iLevel = level
  if (sharpness > 0) {
    if (sharpness > 4) iLevel >>= 2
    else iLevel >>= 1
    if (iLevel > 9 - sharpness) iLevel = 9 - sharpness
  }
  if (iLevel < 1) iLevel = 1
  const limit = 2 * level + iLevel
  const hevThresh = level >= 40 ? 2 : level >= 15 ? 1 : 0
  return { iLevel, limit, hevThresh, inner: i4x4 }
}
