/**
 * VP8 in-loop deblocking filter.
 *
 * Two flavours:
 *   - "Simple" filter (filter_type == 1): applied across macroblock and
 *     sub-block edges with a thin 4-tap kernel.
 *   - "Normal" filter (filter_type == 0): a wider, conditionally-tapped
 *     kernel that includes the high-edge-variance test and a stronger
 *     8-pixel kernel for macroblock edges.
 *
 * The filter operates on the reconstructed YUV planes after intra
 * prediction + residual addition. It modifies pixels straddling each
 * edge between adjacent 4×4 (sub-block) or 16×16 (macroblock) blocks.
 *
 * Reference: RFC 6386 §15.
 */

import { clip255 } from './idct'

/**
 * Apply the loop filter across a horizontal edge inside a plane. The
 * edge runs along `y = edgeY`, separating row `edgeY-1` and row `edgeY`.
 *
 *   - `mbEdge`: true if this is a macroblock-edge filter (uses 8 taps),
 *     false for a sub-block edge (4 taps).
 *   - `nLines`: number of pixels along the edge (typically 16 for a
 *     macroblock edge, 8 for a sub-block edge of a chroma plane, etc.).
 *   - `simple`: true for filter_type == 1.
 *   - `mbLim`/`bLim`/`thresh`: edge thresholds (precomputed for the MB).
 */
export function loopFilterH(
  plane: Uint8Array, stride: number, edgeY: number,
  startX: number, nLines: number,
  mbEdge: boolean, simple: boolean,
  mbLim: number, bLim: number, thresh: number,
): void {
  for (let i = 0; i < nLines; i++) {
    const x = startX + i
    // Sample 8 pixels straddling the edge: p3..p0 (above) and q0..q3 (below).
    const idx0 = (edgeY - 4) * stride + x
    const p3 = plane[idx0 + 0 * stride]
    const p2 = plane[idx0 + 1 * stride]
    const p1 = plane[idx0 + 2 * stride]
    const p0 = plane[idx0 + 3 * stride]
    const q0 = plane[idx0 + 4 * stride]
    const q1 = plane[idx0 + 5 * stride]
    const q2 = plane[idx0 + 6 * stride]
    const q3 = plane[idx0 + 7 * stride]

    if (simple) {
      const filt = simpleFilter(p1, p0, q0, q1, mbLim)
      if (filt === null) continue
      plane[idx0 + 3 * stride] = filt.p0
      plane[idx0 + 4 * stride] = filt.q0
      continue
    }

    const limit = mbEdge ? mbLim : bLim
    if (!normalNeedsFiltering(p3, p2, p1, p0, q0, q1, q2, q3, limit)) continue
    const hev = isHighEdgeVariance(p1, p0, q0, q1, thresh)

    if (mbEdge) {
      const r = mbFilter(p2, p1, p0, q0, q1, q2, hev)
      plane[idx0 + 1 * stride] = r.p2
      plane[idx0 + 2 * stride] = r.p1
      plane[idx0 + 3 * stride] = r.p0
      plane[idx0 + 4 * stride] = r.q0
      plane[idx0 + 5 * stride] = r.q1
      plane[idx0 + 6 * stride] = r.q2
    }
    else {
      const r = bFilter(p1, p0, q0, q1, hev)
      plane[idx0 + 2 * stride] = r.p1
      plane[idx0 + 3 * stride] = r.p0
      plane[idx0 + 4 * stride] = r.q0
      plane[idx0 + 5 * stride] = r.q1
    }
  }
}

/** Mirror of `loopFilterH` for vertical edges (between columns). */
export function loopFilterV(
  plane: Uint8Array, stride: number, edgeX: number,
  startY: number, nLines: number,
  mbEdge: boolean, simple: boolean,
  mbLim: number, bLim: number, thresh: number,
): void {
  for (let i = 0; i < nLines; i++) {
    const y = startY + i
    const base = y * stride + edgeX
    const p3 = plane[base - 4]
    const p2 = plane[base - 3]
    const p1 = plane[base - 2]
    const p0 = plane[base - 1]
    const q0 = plane[base + 0]
    const q1 = plane[base + 1]
    const q2 = plane[base + 2]
    const q3 = plane[base + 3]

    if (simple) {
      const filt = simpleFilter(p1, p0, q0, q1, mbLim)
      if (filt === null) continue
      plane[base - 1] = filt.p0
      plane[base + 0] = filt.q0
      continue
    }

    const limit = mbEdge ? mbLim : bLim
    if (!normalNeedsFiltering(p3, p2, p1, p0, q0, q1, q2, q3, limit)) continue
    const hev = isHighEdgeVariance(p1, p0, q0, q1, thresh)

    if (mbEdge) {
      const r = mbFilter(p2, p1, p0, q0, q1, q2, hev)
      plane[base - 3] = r.p2
      plane[base - 2] = r.p1
      plane[base - 1] = r.p0
      plane[base + 0] = r.q0
      plane[base + 1] = r.q1
      plane[base + 2] = r.q2
    }
    else {
      const r = bFilter(p1, p0, q0, q1, hev)
      plane[base - 2] = r.p1
      plane[base - 1] = r.p0
      plane[base + 0] = r.q0
      plane[base + 1] = r.q1
    }
  }
}

// ---------------------------------------------------------------------------
// Filter primitives — direct ports of RFC 6386 §15
// ---------------------------------------------------------------------------

function clip128(v: number): number {
  v = v - 128
  if (v < -128) v = -128
  if (v > 127) v = 127
  return v
}

function normalNeedsFiltering(
  p3: number, p2: number, p1: number, p0: number,
  q0: number, q1: number, q2: number, q3: number,
  limit: number,
): boolean {
  return Math.abs(p0 - q0) * 2 + (Math.abs(p1 - q1) >> 1) <= limit
    && Math.abs(p3 - p2) <= limit
    && Math.abs(p2 - p1) <= limit
    && Math.abs(p1 - p0) <= limit
    && Math.abs(q3 - q2) <= limit
    && Math.abs(q2 - q1) <= limit
    && Math.abs(q1 - q0) <= limit
}

function isHighEdgeVariance(p1: number, p0: number, q0: number, q1: number, thresh: number): boolean {
  return Math.abs(p1 - p0) > thresh || Math.abs(q1 - q0) > thresh
}

interface SimpleResult { p0: number, q0: number }

function simpleFilter(p1: number, p0: number, q0: number, q1: number, limit: number): SimpleResult | null {
  // Simple-edge test.
  if (Math.abs(p0 - q0) * 2 + (Math.abs(p1 - q1) >> 1) > limit) return null
  let f = clip128(clip128(p1 - q1) + 3 * (q0 - p0))
  const a = clip128((f + 4) >> 3) // wait — (clip128(f+4) >> 3)? See spec.
  // Per RFC 6386 §15.2: f1 = clamp((f + 4) >> 3); f2 = clamp((f + 3) >> 3).
  const f1 = clipSigned7((f + 4) >> 3)
  const f2 = clipSigned7((f + 3) >> 3)
  void a
  return {
    q0: clip255(q0 - f1),
    p0: clip255(p0 + f2),
  }
}

interface BResult { p1: number, p0: number, q0: number, q1: number }
interface MBResult extends BResult { p2: number, q2: number }

function bFilter(p1: number, p0: number, q0: number, q1: number, hev: boolean): BResult {
  const fp1 = clip128(p1)
  const fq1 = clip128(q1)
  const fp0 = clip128(p0)
  const fq0 = clip128(q0)
  let f = hev ? clipSigned7(fp1 - fq1) : 0
  f = clipSigned7(f + 3 * (fq0 - fp0))
  const f1 = clipSigned7((f + 4) >> 3)
  const f2 = clipSigned7((f + 3) >> 3)
  const newP0 = clip255(p0 + f2)
  const newQ0 = clip255(q0 - f1)
  let newP1 = p1
  let newQ1 = q1
  if (!hev) {
    const a = clipSigned7((f1 + 1) >> 1)
    newP1 = clip255(p1 + a)
    newQ1 = clip255(q1 - a)
  }
  return { p1: newP1, p0: newP0, q0: newQ0, q1: newQ1 }
}

function mbFilter(p2: number, p1: number, p0: number, q0: number, q1: number, q2: number, hev: boolean): MBResult {
  if (hev) {
    // Falls back to b-filter shape for pixels closer to the edge.
    const fp1 = clip128(p1)
    const fq1 = clip128(q1)
    const fp0 = clip128(p0)
    const fq0 = clip128(q0)
    let f = clipSigned7(fp1 - fq1)
    f = clipSigned7(f + 3 * (fq0 - fp0))
    const f1 = clipSigned7((f + 4) >> 3)
    const f2 = clipSigned7((f + 3) >> 3)
    return {
      p2,
      p1,
      p0: clip255(p0 + f2),
      q0: clip255(q0 - f1),
      q1,
      q2,
    }
  }
  const fp1 = clip128(p1)
  const fp0 = clip128(p0)
  const fq0 = clip128(q0)
  const fq1 = clip128(q1)
  const w = clipSigned7(clipSigned7(fp1 - fq1) + 3 * (fq0 - fp0))
  // Wider 7-tap kernel for non-HEV macroblock edges.
  const a = (27 * w + 63) >> 7
  const b = (18 * w + 63) >> 7
  const c = (9 * w + 63) >> 7
  return {
    p2: clip255(p2 + clipSigned7(c)),
    p1: clip255(p1 + clipSigned7(b)),
    p0: clip255(p0 + clipSigned7(a)),
    q0: clip255(q0 - clipSigned7(a)),
    q1: clip255(q1 - clipSigned7(b)),
    q2: clip255(q2 - clipSigned7(c)),
  }
}

function clipSigned7(v: number): number {
  if (v < -128) v = -128
  if (v > 127) v = 127
  return v
}

// ---------------------------------------------------------------------------
// Threshold derivation — RFC 6386 §15.4
// ---------------------------------------------------------------------------

/**
 * Compute the (mbLim, bLim, thresh) triple from the raw `loop_filter_level`
 * value. These are the thresholds passed to `loopFilterH`/`loopFilterV`.
 *
 *   mbLim  = 2 * (level + 2) + bLim
 *   bLim   = level + (level >= 40 ? 4 : level >= 15 ? 3 : level >= 1 ? 2 : 1)
 *   thresh = level >= 40 ? 2 : level >= 15 ? 1 : 0
 */
export function deriveLimits(level: number): { mbLim: number, bLim: number, thresh: number } {
  if (level === 0) return { mbLim: 0, bLim: 0, thresh: 0 }
  let bLim = level
  if (level >= 40) bLim += 4
  else if (level >= 15) bLim += 3
  else bLim += 2
  const mbLim = 2 * (level + 2) + bLim
  let thresh: number
  if (level >= 40) thresh = 2
  else if (level >= 15) thresh = 1
  else thresh = 0
  return { mbLim, bLim, thresh }
}
