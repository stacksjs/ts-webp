/**
 * VP8 (lossy WebP) decoder — full pixel decode.
 *
 * Pipeline:
 *   1. Parse the frame header (start code, dimensions, quant indices,
 *      filter params, coef probs) — `header.ts`.
 *   2. Decode all macroblock mode-info from the first partition
 *      (Y mode, B-modes if B_PRED, UV mode, skip flag) — `mode.ts`.
 *   3. For each macroblock, decode coefficient tokens from the token
 *      partition(s) (`coeff.ts`), inverse-transform (`idct.ts`), and
 *      add the residual onto the intra-predicted patch (`intra.ts`).
 *   4. Apply the in-loop deblocking filter to the reconstructed YUV
 *      planes (`loop-filter.ts`).
 *   5. Convert YUV 4:2:0 → RGBA.
 */
import type { WebpImageData } from '../types'
import { BoolDecoder } from './bool-decoder'
import { BLOCK_TYPE_UV, BLOCK_TYPE_Y2, BLOCK_TYPE_Y_AFTER_Y2, BLOCK_TYPE_Y_NO_Y2, decodeBlockCoeffs } from './coeff'
import { parseVP8Header } from './header'
import { idct4x4Add, iwht4x4 } from './idct'
import { predict16x16, predictB, predictChroma8x8 } from './intra'
import { deriveLimits, loopFilterH, loopFilterV } from './loop-filter'
import {
  B_DC_PRED, B_PRED, decodeBPredModes, decodeKeyframeUVMode, decodeKeyframeYMode,
  impliedBmode,
} from './mode'
import {
  DC_PRED, UV_AC_QUANT, UV_DC_QUANT, Y2_AC_QUANT, Y2_DC_QUANT, Y_AC_QUANT, Y_DC_QUANT,
  ZIGZAG, clampQi,
} from './tables'

// Scratch buffers held at module scope so we don't re-allocate per MB.
// Hoisted before everything else so the linter sees them as used by
// the functions below.
const stagedYResidual = new Int16Array(16 * 16) // 16 Y blocks × 16 coefs each
const stagedUResidual = new Int16Array(4 * 16)
const stagedVResidual = new Int16Array(4 * 16)

interface MacroblockInfo {
  yMode: number
  uvMode: number
  bmodes: Uint8Array // 16 entries, only meaningful when yMode === B_PRED
  skip: boolean // mb_skip_coef — true if all coefficients are zero for this MB
  nonZeroY: number // bitmap (16 bits) of which Y blocks have non-zero coefs
  nonZeroUV: number // bitmap (8 bits) of which UV blocks (4U+4V) have non-zero coefs
}

export function decodeVP8(data: Uint8Array): WebpImageData {
  const header = parseVP8Header(data)
  const W = header.frame.width
  const H = header.frame.height

  // Round to macroblock multiples — VP8 always processes whole MBs.
  const mbW = (W + 15) >> 4
  const mbH = (H + 15) >> 4
  const yStride = mbW * 16
  const uvStride = mbW * 8

  // Reconstruction buffers (will be cropped to W×H at the end).
  const yPlane = new Uint8Array(yStride * mbH * 16)
  const uPlane = new Uint8Array(uvStride * mbH * 8)
  const vPlane = new Uint8Array(uvStride * mbH * 8)

  // Compute dequantisation factors.
  const qi = clampQi(header.quantiser.yacQi)
  const yAc = Y_AC_QUANT[clampQi(qi + header.quantiser.y2acDelta)] // not used for Y; AC of Y is yAcN
  void yAc
  const dq = computeDequant(header.quantiser)

  // ----- Phase 1: decode all mode info from the first partition. -----
  const mbInfo: MacroblockInfo[] = new Array(mbW * mbH)
  // Above/left mode tracking for B_PRED contextual coding.
  const aboveYModes = new Uint8Array(mbW * 4) // 4 sub-blocks per MB column
  const leftYModes = new Uint8Array(4)

  for (let mbY = 0; mbY < mbH; mbY++) {
    for (let i = 0; i < 4; i++) leftYModes[i] = B_DC_PRED
    for (let mbX = 0; mbX < mbW; mbX++) {
      const skip = header.probSkipFalse !== null
        ? header.bool.readBit(header.probSkipFalse) === 1
        : false
      const yMode = decodeKeyframeYMode(header.bool)
      const bmodes = new Uint8Array(16)
      if (yMode === B_PRED) {
        const above4 = new Uint8Array(4)
        for (let i = 0; i < 4; i++) above4[i] = aboveYModes[mbX * 4 + i]
        decodeBPredModes(header.bool, above4, leftYModes, bmodes)
        for (let i = 0; i < 4; i++) aboveYModes[mbX * 4 + i] = above4[i]
      }
      else {
        // Non-B_PRED MB seeds neighbour-mode arrays with the implied B-mode.
        const im = impliedBmode(yMode)
        for (let i = 0; i < 4; i++) {
          aboveYModes[mbX * 4 + i] = im
          leftYModes[i] = im
        }
        for (let i = 0; i < 16; i++) bmodes[i] = im
      }
      const uvMode = decodeKeyframeUVMode(header.bool)
      mbInfo[mbY * mbW + mbX] = {
        yMode,
        uvMode,
        bmodes,
        skip,
        nonZeroY: 0,
        nonZeroUV: 0,
      }
    }
  }

  // ----- Phase 2: open the residual (token) partition. -----
  // For numPartitions == 1 the residual data is one contiguous partition
  // starting at `partitionsOffset`. For more partitions there's a small
  // index of (numPartitions-1) 3-byte sizes preceding them; macroblock
  // rows are striped across partitions modulo numPartitions.
  let resBool: BoolDecoder
  if (header.numPartitions === 1) {
    resBool = new BoolDecoder(data, header.partitionsOffset)
  }
  else {
    // Multi-partition support: parse the size index and pick partition[0].
    // We currently only handle single-partition WebP files (which is
    // what cwebp produces by default for small images). Throw a clear
    // error if we hit the multi-partition path.
    throw new Error('VP8: multi-partition residuals are not implemented yet')
  }

  // ----- Phase 3: decode coefficients + reconstruct each macroblock. -----
  // Per-column persistent context for "previous block had non-zero coefs?"
  // — used as the (above) context for coefficient decoding.
  const aboveNzY = new Uint8Array(mbW * 4)
  const aboveNzU = new Uint8Array(mbW * 2)
  const aboveNzV = new Uint8Array(mbW * 2)
  const aboveNzY2 = new Uint8Array(mbW)

  const coeffs = new Int16Array(16) // scratch
  const yCoeffsRowMajor = new Int16Array(16)

  for (let mbY = 0; mbY < mbH; mbY++) {
    const leftNzY = new Uint8Array(4)
    const leftNzU = new Uint8Array(2)
    const leftNzV = new Uint8Array(2)
    let leftNzY2 = 0

    for (let mbX = 0; mbX < mbW; mbX++) {
      const info = mbInfo[mbY * mbW + mbX]
      const has16 = info.yMode !== B_PRED // 16×16-prediction MB uses the Y2 block

      if (info.skip) {
        // All blocks treated as zero-coef.
        if (has16) {
          aboveNzY2[mbX] = 0
          leftNzY2 = 0
        }
        for (let i = 0; i < 4; i++) {
          aboveNzY[mbX * 4 + i] = 0
          leftNzY[i] = 0
        }
        for (let i = 0; i < 2; i++) {
          aboveNzU[mbX * 2 + i] = 0
          leftNzU[i] = 0
          aboveNzV[mbX * 2 + i] = 0
          leftNzV[i] = 0
        }
        info.nonZeroY = 0
        info.nonZeroUV = 0
      }
      else {
        // Y2 block (when 16×16 mode).
        let y2Coefs: Int16Array | null = null
        if (has16) {
          y2Coefs = new Int16Array(16)
          const nz = decodeBlockCoeffs(resBool, y2Coefs, BLOCK_TYPE_Y2, leftNzY2, aboveNzY2[mbX], 0, header.coefProbs)
          aboveNzY2[mbX] = nz
          leftNzY2 = nz
          // Dequantise Y2 — DC0 with y2Dc, rest with y2Ac.
          y2Coefs[0] *= dq.y2Dc
          for (let i = 1; i < 16; i++) y2Coefs[i] *= dq.y2Ac
          iwht4x4(y2Coefs)
        }

        // 16 Y blocks.
        let yNZ = 0
        for (let by = 0; by < 4; by++) {
          for (let bx = 0; bx < 4; bx++) {
            const subIdx = by * 4 + bx
            const blockType = has16 ? BLOCK_TYPE_Y_AFTER_Y2 : BLOCK_TYPE_Y_NO_Y2
            const firstCoef = has16 ? 1 : 0
            const nz = decodeBlockCoeffs(
              resBool, coeffs, blockType,
              leftNzY[by], aboveNzY[mbX * 4 + bx],
              firstCoef, header.coefProbs,
            )
            aboveNzY[mbX * 4 + bx] = nz
            leftNzY[by] = nz
            if (nz) yNZ |= 1 << subIdx
            // Inject Y2 DC if 16×16 mode.
            if (has16) coeffs[0] = y2Coefs![subIdx]
            // Dequantise.
            coeffs[0] *= dq.yDc
            for (let i = 1; i < 16; i++) coeffs[i] *= dq.yAc
            // Un-zigzag into row-major raster order.
            for (let i = 0; i < 16; i++) yCoeffsRowMajor[ZIGZAG[i]] = coeffs[i]
            // Add residual onto the predicted Y patch (already in plane).
            // Prediction is performed below in a two-pass scheme:
            // first we predict the whole MB, then add residuals per
            // sub-block. But for B_PRED we have to interleave per-block.
            // Stage residual aside for later in this MB.
            stagedYResidual.set(yCoeffsRowMajor, subIdx * 16)
          }
        }
        info.nonZeroY = yNZ

        // 4 U blocks then 4 V blocks.
        let uvNZ = 0
        for (let plane = 0; plane < 2; plane++) {
          const aboveNz = plane === 0 ? aboveNzU : aboveNzV
          const leftNz = plane === 0 ? leftNzU : leftNzV
          for (let by = 0; by < 2; by++) {
            for (let bx = 0; bx < 2; bx++) {
              const subIdx = by * 2 + bx
              const nz = decodeBlockCoeffs(
                resBool, coeffs, BLOCK_TYPE_UV,
                leftNz[by], aboveNz[mbX * 2 + bx],
                0, header.coefProbs,
              )
              aboveNz[mbX * 2 + bx] = nz
              leftNz[by] = nz
              if (nz) uvNZ |= 1 << (plane * 4 + subIdx)
              coeffs[0] *= dq.uvDc
              for (let i = 1; i < 16; i++) coeffs[i] *= dq.uvAc
              const dst = plane === 0 ? stagedUResidual : stagedVResidual
              const target = new Int16Array(16)
              for (let i = 0; i < 16; i++) target[ZIGZAG[i]] = coeffs[i]
              dst.set(target, subIdx * 16)
            }
          }
        }
        info.nonZeroUV = uvNZ
      }

      // ----- Reconstruct (predict + add residual) the macroblock. -----
      reconstructMB(
        info, mbX, mbY, mbW, mbH,
        yPlane, uPlane, vPlane,
        yStride, uvStride,
        stagedYResidual, stagedUResidual, stagedVResidual,
      )
    }
  }

  // ----- Phase 4: loop filter. -----
  if (header.filter.level > 0) {
    runLoopFilter(
      yPlane, uPlane, vPlane,
      yStride, uvStride,
      mbW, mbH, mbInfo,
      header.filter.level, header.filter.sharpness,
      header.filter.filterType === 1,
    )
  }

  // ----- Phase 5: YUV 4:2:0 → RGBA. -----
  const out = new Uint8Array(W * H * 4)
  yuv420ToRgba(yPlane, uPlane, vPlane, yStride, uvStride, W, H, out)

  return { data: out, width: W, height: H, hasAlpha: false }
}

function computeDequant(q: { yacQi: number, ydcDelta: number, y2dcDelta: number, y2acDelta: number, uvdcDelta: number, uvacDelta: number }): {
  yDc: number, yAc: number, y2Dc: number, y2Ac: number, uvDc: number, uvAc: number,
} {
  const qi = clampQi(q.yacQi)
  return {
    yDc: Y_DC_QUANT[clampQi(qi + q.ydcDelta)],
    yAc: Y_AC_QUANT[qi],
    y2Dc: Math.min(Y2_DC_QUANT[clampQi(qi + q.y2dcDelta)] * 2, 0xFFFF), // RFC 6386 §14.1: ×2
    y2Ac: Math.max(Math.floor(Y2_AC_QUANT[clampQi(qi + q.y2acDelta)] * 155 / 100), 8),
    uvDc: Math.min(UV_DC_QUANT[clampQi(qi + q.uvdcDelta)], 132),
    uvAc: UV_AC_QUANT[clampQi(qi + q.uvacDelta)],
  }
}

/**
 * Predict + add residual for a single macroblock. Sub-block prediction
 * for B_PRED MBs interleaves prediction and residual addition because
 * later sub-blocks depend on earlier reconstructed pixels.
 */
function reconstructMB(
  info: MacroblockInfo, mbX: number, mbY: number, mbW: number, _mbH: number,
  yPlane: Uint8Array, uPlane: Uint8Array, vPlane: Uint8Array,
  yStride: number, uvStride: number,
  yRes: Int16Array, uRes: Int16Array, vRes: Int16Array,
): void {
  const yOff = mbY * 16 * yStride + mbX * 16
  const uOff = mbY * 8 * uvStride + mbX * 8
  const vOff = mbY * 8 * uvStride + mbX * 8

  // ----- Luma -----
  if (info.yMode !== B_PRED) {
    const top = collectTopRow(yPlane, yStride, mbY, mbX, 16)
    const left = collectLeftCol(yPlane, yStride, mbY, mbX, 16)
    const tl = mbY > 0 && mbX > 0 ? yPlane[yOff - yStride - 1] : (mbY === 0 ? 127 : 129)
    predict16x16(info.yMode, yPlane, yStride, yOff, top, left, tl, mbY > 0, mbX > 0)
    // Add residual for each of the 16 Y blocks.
    for (let by = 0; by < 4; by++) {
      for (let bx = 0; bx < 4; bx++) {
        const subIdx = by * 4 + bx
        const subRes = yRes.subarray(subIdx * 16, subIdx * 16 + 16)
        if ((info.nonZeroY >> subIdx) & 1 || !info.skip) {
          idct4x4Add(subRes as Int16Array, yPlane, yStride, yOff + by * 4 * yStride + bx * 4)
        }
      }
    }
  }
  else {
    // B_PRED — sub-block-by-sub-block.
    for (let by = 0; by < 4; by++) {
      for (let bx = 0; bx < 4; bx++) {
        const subIdx = by * 4 + bx
        const subOff = yOff + by * 4 * yStride + bx * 4
        // Build the 8-pixel top vector (above + above-right). Above-right
        // for the rightmost sub-block of a row borrows from the macroblock-
        // edge spec: the 4 above-right pixels are replicated from the last
        // available pixel.
        const top = new Uint8Array(8)
        const topRowAvail = mbY > 0 || by > 0
        if (topRowAvail) {
          for (let i = 0; i < 4; i++) top[i] = yPlane[subOff - yStride + i]
          // Above-right: spec rule. For the top-right sub-block of a
          // macroblock that's NOT in the top row, pixels above-right
          // beyond the MB don't exist; we replicate the rightmost
          // pixel of the above row's MB (top[3]).
          if (bx < 3 || (mbY > 0 && (mbX < mbW - 1 || by > 0))) {
            for (let i = 0; i < 4; i++) {
              const x = bx * 4 + 4 + i
              if (x < 16 && (mbY > 0 || by > 0)) {
                top[4 + i] = yPlane[subOff - yStride + 4 + i]
              }
              else {
                top[4 + i] = top[3]
              }
            }
          }
          else {
            for (let i = 0; i < 4; i++) top[4 + i] = top[3]
          }
        }
        else {
          for (let i = 0; i < 8; i++) top[i] = 127
        }
        const left = new Uint8Array(4)
        const leftAvail = mbX > 0 || bx > 0
        if (leftAvail) {
          for (let i = 0; i < 4; i++) left[i] = yPlane[subOff + i * yStride - 1]
        }
        else {
          for (let i = 0; i < 4; i++) left[i] = 129
        }
        let tl: number
        if (topRowAvail && leftAvail) tl = yPlane[subOff - yStride - 1]
        else if (topRowAvail) tl = 129
        else tl = 127
        predictB(info.bmodes[subIdx], yPlane, yStride, subOff, top, left, tl)
        if ((info.nonZeroY >> subIdx) & 1 || !info.skip) {
          const subRes = yRes.subarray(subIdx * 16, subIdx * 16 + 16) as Int16Array
          idct4x4Add(subRes, yPlane, yStride, subOff)
        }
      }
    }
  }

  // ----- Chroma -----
  for (let plane = 0; plane < 2; plane++) {
    const buf = plane === 0 ? uPlane : vPlane
    const off = plane === 0 ? uOff : vOff
    const top = collectTopRow(buf, uvStride, mbY, mbX, 8)
    const left = collectLeftCol(buf, uvStride, mbY, mbX, 8)
    const tl = mbY > 0 && mbX > 0 ? buf[off - uvStride - 1] : (mbY === 0 ? 127 : 129)
    predictChroma8x8(info.uvMode, buf, uvStride, off, top, left, tl, mbY > 0, mbX > 0)
    const res = plane === 0 ? uRes : vRes
    for (let by = 0; by < 2; by++) {
      for (let bx = 0; bx < 2; bx++) {
        const subIdx = by * 2 + bx
        const subRes = res.subarray(subIdx * 16, subIdx * 16 + 16) as Int16Array
        const flagBit = plane === 0 ? subIdx : 4 + subIdx
        if ((info.nonZeroUV >> flagBit) & 1 || !info.skip) {
          idct4x4Add(subRes, buf, uvStride, off + by * 4 * uvStride + bx * 4)
        }
      }
    }
  }
}

function collectTopRow(plane: Uint8Array, stride: number, mbY: number, mbX: number, n: number): Uint8Array {
  const out = new Uint8Array(n)
  if (mbY === 0) {
    out.fill(127)
  }
  else {
    const baseY = mbY * n - 1
    for (let i = 0; i < n; i++) out[i] = plane[baseY * stride + mbX * n + i]
  }
  return out
}

function collectLeftCol(plane: Uint8Array, stride: number, mbY: number, mbX: number, n: number): Uint8Array {
  const out = new Uint8Array(n)
  if (mbX === 0) {
    out.fill(129)
  }
  else {
    const baseX = mbX * n - 1
    for (let i = 0; i < n; i++) out[i] = plane[(mbY * n + i) * stride + baseX]
  }
  return out
}

/**
 * Apply the loop filter across all macroblock and sub-block edges.
 *
 * For each macroblock:
 *   - Filter the 4 vertical sub-block edges within the MB (skipping the
 *     leftmost MB column where only the MB-edge filter applies).
 *   - Filter the 4 horizontal sub-block edges within the MB.
 *   - Filter the MB's left edge against the previous MB (mbEdge=true).
 *   - Filter the MB's top edge against the MB above (mbEdge=true).
 *
 * The macroblock filter level may be modulated by the segment id and
 * by per-mode/per-ref deltas, but for our minimal keyframe path we use
 * the base level directly.
 */
function runLoopFilter(
  yPlane: Uint8Array, uPlane: Uint8Array, vPlane: Uint8Array,
  yStride: number, uvStride: number,
  mbW: number, mbH: number, mbInfo: MacroblockInfo[],
  baseLevel: number, _sharpness: number, simple: boolean,
): void {
  const lim = deriveLimits(baseLevel)

  for (let mbY = 0; mbY < mbH; mbY++) {
    for (let mbX = 0; mbX < mbW; mbX++) {
      const info = mbInfo[mbY * mbW + mbX]
      // Skip filtering if the MB has no non-zero coefs and uses a
      // "no transform" prediction (DC_PRED with no residual).
      // Simple optimization — we always filter for now to stay correct.
      void info

      // Vertical edges within the MB (between sub-block columns).
      for (let i = 1; i < 4; i++) {
        const x = mbX * 16 + i * 4
        loopFilterV(yPlane, yStride, x, mbY * 16, 16, false, simple, lim.mbLim, lim.bLim, lim.thresh)
      }
      // Horizontal edges within the MB.
      for (let i = 1; i < 4; i++) {
        const y = mbY * 16 + i * 4
        loopFilterH(yPlane, yStride, y, mbX * 16, 16, false, simple, lim.mbLim, lim.bLim, lim.thresh)
      }
      // Vertical MB edge (left).
      if (mbX > 0) {
        loopFilterV(yPlane, yStride, mbX * 16, mbY * 16, 16, true, simple, lim.mbLim, lim.bLim, lim.thresh)
      }
      // Horizontal MB edge (top).
      if (mbY > 0) {
        loopFilterH(yPlane, yStride, mbY * 16, mbX * 16, 16, true, simple, lim.mbLim, lim.bLim, lim.thresh)
      }

      if (!simple) {
        // Chroma — only the mid-MB edge (one per axis, at offset 4).
        for (const plane of [uPlane, vPlane]) {
          loopFilterV(plane, uvStride, mbX * 8 + 4, mbY * 8, 8, false, false, lim.mbLim, lim.bLim, lim.thresh)
          loopFilterH(plane, uvStride, mbY * 8 + 4, mbX * 8, 8, false, false, lim.mbLim, lim.bLim, lim.thresh)
          if (mbX > 0)
            loopFilterV(plane, uvStride, mbX * 8, mbY * 8, 8, true, false, lim.mbLim, lim.bLim, lim.thresh)
          if (mbY > 0)
            loopFilterH(plane, uvStride, mbY * 8, mbX * 8, 8, true, false, lim.mbLim, lim.bLim, lim.thresh)
        }
      }
    }
  }
}

/**
 * BT.601 YUV 4:2:0 → RGBA. Each chroma sample covers a 2×2 luma quad.
 * Coefficients are the libvpx fixed-point ones (shift 14):
 *   R = Y + 1.402  * (V - 128)
 *   G = Y - 0.344  * (U - 128) - 0.714 * (V - 128)
 *   B = Y + 1.772  * (U - 128)
 */
function yuv420ToRgba(
  Y: Uint8Array, U: Uint8Array, V: Uint8Array,
  yStride: number, uvStride: number,
  W: number, H: number, out: Uint8Array,
): void {
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const yi = Y[y * yStride + x]
      const ui = U[(y >> 1) * uvStride + (x >> 1)] - 128
      const vi = V[(y >> 1) * uvStride + (x >> 1)] - 128
      const r = clip255(yi + ((22970 * vi + 8192) >> 14))
      const g = clip255(yi - ((5638 * ui + 11700 * vi + 8192) >> 14))
      const b = clip255(yi + ((29032 * ui + 8192) >> 14))
      const o = (y * W + x) * 4
      out[o] = r
      out[o + 1] = g
      out[o + 2] = b
      out[o + 3] = 255
    }
  }
}

function clip255(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v
}
