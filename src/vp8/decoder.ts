/**
 * VP8 (lossy WebP) decoder — full pixel decode.
 *
 * Top-down port of libwebp's `ParseFrame`/`VP8DecodeMB`/`ReconstructRow`/
 * `FilterRow` pipeline (src/dec/vp8_dec.c + frame_dec.c). Each macroblock
 * row is processed in three passes:
 *   1. `parseIntraModeRow` — decode mode info for every MB in the row from
 *      the first-partition bool reader.
 *   2. `decodeMB` — decode each MB's residual coefficients from the
 *      relevant token partition.
 *   3. `reconstructRow` + `filterRow` — predict, add residuals, and run
 *      the in-loop deblocking filter.
 *
 * The per-MB working buffer (`yuvB`) holds Y/U/V at fixed offsets with one
 * row of "above" context and one column of "left" context surrounding it,
 * so the predictors can read neighbour pixels via simple offsets
 * (`dst[off - BPS]`, `dst[off - 1]`, etc.). After reconstruction the patch
 * is copied into the cache (`cacheY`/`cacheU`/`cacheV`), filtering happens
 * on the cache, and at the end the cache is converted to RGBA.
 */
import type { WebpImageData } from '../types'
import type { VP8QuantMatrix } from './header'
import { BoolDecoder } from './bool-decoder'
import {
  BLOCK_TYPE_UV, BLOCK_TYPE_Y2, BLOCK_TYPE_Y_AFTER_Y2, BLOCK_TYPE_Y_NO_Y2,
  decodeBlockCoeffs,
} from './coeff'
import { parseVP8Header } from './header'
import { idct4x4Add, iwht4x4, transformDC } from './idct'
import {
  B_DC_PRED_NOLEFT, B_DC_PRED_NOTOP, B_DC_PRED_NOTOPLEFT, BPS,
  predictChroma8, predictLuma16, predictLuma4,
} from './intra'
import {
  hFilter16, hFilter16i, hFilter8, hFilter8i, precomputeFilterStrength,
  simpleHFilter16, simpleHFilter16i, simpleVFilter16, simpleVFilter16i,
  vFilter16, vFilter16i, vFilter8, vFilter8i,
} from './loop-filter'
import type { MBData } from './mode'
import { parseIntraMode } from './mode'
import { B_DC_PRED, B_PRED, DC_PRED } from './tables'

// Per-macroblock buffer layout. Same as libwebp's `yuv_b` (128 bytes
// total when summed across YUV planes, padded to alignment in C).
//
//   Y at offset Y_OFF (16×16, with 1-pixel left margin and 1-row top margin)
//   U at offset U_OFF (8×8, similarly)
//   V at offset V_OFF
//
// We use a single Uint8Array large enough to hold all three with their
// surrounding context.
const Y_OFF = BPS + 8 // libwebp: BPS + 8
const U_OFF = Y_OFF + BPS * 16 + BPS // skip past Y patch + a separator row
const V_OFF = U_OFF + 16 // 8 bytes after U on the same rows
const YUV_SIZE = V_OFF + BPS * 8

/** kScan: zig-zag-of-4×4-blocks position → byte offset within `yuvB[Y_OFF..]`. */
const K_SCAN = new Uint16Array([
  0 + 0 * BPS, 4 + 0 * BPS, 8 + 0 * BPS, 12 + 0 * BPS,
  0 + 4 * BPS, 4 + 4 * BPS, 8 + 4 * BPS, 12 + 4 * BPS,
  0 + 8 * BPS, 4 + 8 * BPS, 8 + 8 * BPS, 12 + 8 * BPS,
  0 + 12 * BPS, 4 + 12 * BPS, 8 + 12 * BPS, 12 + 12 * BPS,
])

interface MB {
  /** Bit-flag bitmap for "block had any non-zero coef" (libwebp's `nz`). */
  nz: number
  /** Same for DC-only contexts (libwebp's `nz_dc`). */
  nzDc: number
}

/** Pack non-zero coefficient flags as libwebp's `non_zero_y`/`non_zero_uv`. */
function nzCodeBits(nzCoeffs: number, nz: number, dcNz: number): number {
  let out = (nzCoeffs << 2) >>> 0
  out |= nz > 3 ? 3 : nz > 1 ? 2 : dcNz
  return out >>> 0
}

export function decodeVP8(data: Uint8Array): WebpImageData {
  const header = parseVP8Header(data)
  const W = header.frame.width
  const H = header.frame.height
  const mbW = (W + 15) >> 4
  const mbH = (H + 15) >> 4

  // Final image cache (full Y/U/V planes).
  const cacheYStride = mbW * 16
  const cacheUVStride = mbW * 8
  const cacheY = new Uint8Array(cacheYStride * mbH * 16)
  const cacheU = new Uint8Array(cacheUVStride * mbH * 8)
  const cacheV = new Uint8Array(cacheUVStride * mbH * 8)

  // Top-row sample cache (one entry per MB column).
  const yuvT = {
    y: new Uint8Array(mbW * 16),
    u: new Uint8Array(mbW * 8),
    v: new Uint8Array(mbW * 8),
  }

  // Per-MB scratch buffer with surrounding context. Reused across MBs.
  const yuvB = new Uint8Array(YUV_SIZE)

  // intra_t / intra_l mode trackers.
  const intraT = new Uint8Array(mbW * 4) // bottom row of MBs above
  const intraL = new Uint8Array(4) // right column of MB to the left

  // Per-row MB metadata (mode + coefficients) — libwebp keeps just one row.
  const mbData: MBData[] = []
  for (let i = 0; i < mbW; i++) {
    mbData.push({
      segment: 0,
      skip: false,
      isI4x4: false,
      imodes: new Uint8Array(16),
      uvMode: 0,
      coeffs: new Int16Array(384),
      nonZeroY: 0,
      nonZeroUV: 0,
      nonZeroDc: 0,
    })
  }
  const mbInfo: MB[] = []
  for (let i = 0; i < mbW + 1; i++) {
    mbInfo.push({ nz: 0, nzDc: 0 })
  }

  // ── Set up token partitions ─────────────────────────────────────────────
  const tokenPartitions = parseTokenPartitions(data, header.partitionsOffset, header.numPartitions)
  const useSkipProba = header.probSkipFalse !== null
  const skipP = header.probSkipFalse ?? 0

  // Pre-compute filter strengths per (segment, i4x4).
  const fStrengths: Array<[ReturnType<typeof precomputeFilterStrength>, ReturnType<typeof precomputeFilterStrength>]> = []
  for (let s = 0; s < 4; s++) {
    let baseLevel: number
    if (header.segmentation.useSegment) {
      baseLevel = header.segmentation.segments[s].filterStrength
      if (!header.segmentation.absoluteDelta) baseLevel += header.filter.level
    }
    else {
      baseLevel = header.filter.level
    }
    const f0 = precomputeFilterStrength(
      baseLevel, header.filter.useLfDelta,
      header.filter.refLfDelta[0], header.filter.modeLfDelta[0],
      header.filter.sharpness, false,
    )
    const f1 = precomputeFilterStrength(
      baseLevel, header.filter.useLfDelta,
      header.filter.refLfDelta[0], header.filter.modeLfDelta[0],
      header.filter.sharpness, true,
    )
    fStrengths.push([f0, f1])
  }

  // ── Main decoding loop ──────────────────────────────────────────────────
  for (let mbY = 0; mbY < mbH; mbY++) {
    // Reset left-MB tracker for this row.
    intraL.fill(B_DC_PRED)
    mbInfo[0].nz = 0
    mbInfo[0].nzDc = 0

    // Pass 1: parse mode info for every MB in this row.
    for (let mbX = 0; mbX < mbW; mbX++) {
      parseIntraMode(
        header.bool, header.segmentation, useSkipProba, skipP,
        intraT, mbX * 4, intraL, mbData[mbX],
      )
    }

    // Pass 2: decode residuals from the row's token partition.
    const tokenBR = tokenPartitions[mbY % header.numPartitions]
    for (let mbX = 0; mbX < mbW; mbX++) {
      decodeMB(
        tokenBR, mbData[mbX], mbInfo[mbX], mbInfo[mbX + 1],
        useSkipProba, header.coefProbs, header.quant,
      )
    }

    // Pass 3: reconstruct + filter.
    reconstructRow(
      mbY, mbW, mbH, mbData, yuvB, yuvT,
      cacheY, cacheU, cacheV, cacheYStride, cacheUVStride,
    )
    if (header.filter.filterType > 0) {
      filterRow(
        mbY, mbW, mbH, mbData, fStrengths, header.filter.filterType,
        cacheY, cacheU, cacheV, cacheYStride, cacheUVStride,
      )
    }
  }

  // ── YUV → RGBA ──────────────────────────────────────────────────────────
  const out = new Uint8Array(W * H * 4)
  yuv420ToRgba(cacheY, cacheU, cacheV, cacheYStride, cacheUVStride, W, H, out)

  return { data: out, width: W, height: H, hasAlpha: false }
}

// ---------------------------------------------------------------------------
// Token partition setup (multi-partition support — RFC 6386 §9.5)
// ---------------------------------------------------------------------------

function parseTokenPartitions(
  data: Uint8Array, offset: number, numPartitions: number,
): BoolDecoder[] {
  // For numPartitions > 1, the first (numPartitions - 1) × 3 bytes encode
  // little-endian sizes of partitions 0..N-2; the last partition's size is
  // implicit (whatever's left).
  const out: BoolDecoder[] = []
  if (numPartitions === 1) {
    out.push(new BoolDecoder(data, offset, data.length - offset))
    return out
  }
  const sizesEnd = offset + (numPartitions - 1) * 3
  let p = sizesEnd
  for (let i = 0; i < numPartitions - 1; i++) {
    const idx = offset + i * 3
    const size = data[idx] | (data[idx + 1] << 8) | (data[idx + 2] << 16)
    if (p + size > data.length) {
      throw new Error(`VP8: partition ${i} size ${size} overruns buffer`)
    }
    out.push(new BoolDecoder(data, p, size))
    p += size
  }
  // Last partition: everything left.
  out.push(new BoolDecoder(data, p, data.length - p))
  return out
}

// ---------------------------------------------------------------------------
// Per-MB residual decode (port of libwebp's ParseResiduals)
// ---------------------------------------------------------------------------

function decodeMB(
  tokenBR: BoolDecoder, block: MBData, leftMb: MB, mb: MB,
  useSkipProba: boolean, coefProbs: Uint8Array, quant: VP8QuantMatrix[],
): void {
  const skip = useSkipProba ? block.skip : false
  // Reset coefficient buffer.
  block.coeffs.fill(0)
  if (skip) {
    leftMb.nz = mb.nz = 0
    if (!block.isI4x4) leftMb.nzDc = mb.nzDc = 0
    block.nonZeroY = 0
    block.nonZeroUV = 0
    return
  }

  const q = quant[block.segment]
  let dst = 0
  let nonZeroY = 0
  let nonZeroUV = 0
  let firstCoef: number
  let acProbs: number // block_type for AC coefficient decoding

  // Y2 block, if 16×16 mode.
  if (!block.isI4x4) {
    const dcCoeffs = new Int16Array(16)
    const ctx = mb.nzDc + leftMb.nzDc
    const nz = decodeBlockCoeffs(tokenBR, dcCoeffs, BLOCK_TYPE_Y2, ctx, 0, coefProbs, q.y2[0], q.y2[1])
    mb.nzDc = leftMb.nzDc = nz > 0 ? 1 : 0
    if (nz > 1) {
      // Full WHT — distribute DCs into each Y block's coef[0].
      iwht4x4(dcCoeffs)
      for (let i = 0; i < 16; i++) block.coeffs[i * 16] = dcCoeffs[i]
    }
    else {
      // Only DC of the WHT block was non-zero — broadcast (dc + 3) >> 3.
      const dc0 = (dcCoeffs[0] + 3) >> 3
      for (let i = 0; i < 16; i++) block.coeffs[i * 16] = dc0
    }
    firstCoef = 1
    acProbs = BLOCK_TYPE_Y_AFTER_Y2
  }
  else {
    firstCoef = 0
    acProbs = BLOCK_TYPE_Y_NO_Y2
  }

  // 16 Y blocks. tnz/lnz are 8-bit packed flags shifted as we walk
  // raster-order; each new "non-zero?" flag goes into bit 7, and we
  // shift-right between rows / columns. Mirrors libwebp's bookkeeping.
  let tnz = mb.nz & 0x0F
  let lnz = leftMb.nz & 0x0F
  for (let y = 0; y < 4; y++) {
    let l = lnz & 1
    let nzCoeffs = 0
    for (let x = 0; x < 4; x++) {
      const ctx = l + (tnz & 1)
      const nz = decodeBlockCoeffs(
        tokenBR, block.coeffs.subarray(dst, dst + 16),
        acProbs, ctx, firstCoef, coefProbs, q.y1[0], q.y1[1],
      )
      l = nz > firstCoef ? 1 : 0
      tnz = ((tnz >> 1) | (l << 7)) & 0xFF
      const dcNz = block.coeffs[dst] !== 0 ? 1 : 0
      nzCoeffs = nzCodeBits(nzCoeffs, nz, dcNz)
      dst += 16
    }
    tnz >>= 4
    lnz = ((lnz >> 1) | (l << 7)) & 0xFF
    nonZeroY = ((nonZeroY << 8) | nzCoeffs) >>> 0
  }
  let outTNz = tnz & 0x0F
  let outLNz = (lnz >> 4) & 0x0F

  // 8 chroma blocks (4 U + 4 V), shared block_type=BLOCK_TYPE_UV.
  for (let ch = 0; ch < 4; ch += 2) {
    let nzCoeffs = 0
    let tnzCh = (mb.nz >> (4 + ch)) & 0x0F
    let lnzCh = (leftMb.nz >> (4 + ch)) & 0x0F
    for (let y = 0; y < 2; y++) {
      let l = lnzCh & 1
      for (let x = 0; x < 2; x++) {
        const ctx = l + (tnzCh & 1)
        const nz = decodeBlockCoeffs(
          tokenBR, block.coeffs.subarray(dst, dst + 16),
          BLOCK_TYPE_UV, ctx, 0, coefProbs, q.uv[0], q.uv[1],
        )
        l = nz > 0 ? 1 : 0
        tnzCh = ((tnzCh >> 1) | (l << 3)) & 0x0F
        const dcNz = block.coeffs[dst] !== 0 ? 1 : 0
        nzCoeffs = nzCodeBits(nzCoeffs, nz, dcNz)
        dst += 16
      }
      tnzCh >>= 2
      lnzCh = ((lnzCh >> 1) | (l << 5)) & 0xFF
    }
    nonZeroUV = (nonZeroUV | (nzCoeffs << (4 * ch))) >>> 0
    // Pack chroma flags into bits 4..7 (U for ch=0) and 8..11 (V for ch=2).
    outTNz |= ((tnzCh & 0x0F) << 4) << ch
    outLNz |= ((lnzCh & 0xF0)) << ch
  }
  block.nonZeroY = nonZeroY
  block.nonZeroUV = nonZeroUV
  mb.nz = outTNz
  leftMb.nz = outLNz
}

// ---------------------------------------------------------------------------
// Per-row reconstruction (port of libwebp's ReconstructRow)
// ---------------------------------------------------------------------------

function reconstructRow(
  mbY: number, mbW: number, mbH: number, mbData: MBData[],
  yuvB: Uint8Array, yuvT: { y: Uint8Array, u: Uint8Array, v: Uint8Array },
  cacheY: Uint8Array, cacheU: Uint8Array, cacheV: Uint8Array,
  cacheYStride: number, cacheUVStride: number,
): void {
  const yDst = Y_OFF
  const uDst = U_OFF
  const vDst = V_OFF
  void mbH

  // Initialise left-most block's left column to 129 (libwebp sentinel).
  for (let j = 0; j < 16; j++) yuvB[yDst + j * BPS - 1] = 129
  for (let j = 0; j < 8; j++) {
    yuvB[uDst + j * BPS - 1] = 129
    yuvB[vDst + j * BPS - 1] = 129
  }

  if (mbY > 0) {
    yuvB[yDst - 1 - BPS] = 129
    yuvB[uDst - 1 - BPS] = 129
    yuvB[vDst - 1 - BPS] = 129
  }
  else {
    // Top row of frame — fill row above + extra right margin with 127.
    for (let i = -1; i < 16 + 4; i++) yuvB[yDst + i - BPS] = 127
    for (let i = -1; i < 8; i++) {
      yuvB[uDst + i - BPS] = 127
      yuvB[vDst + i - BPS] = 127
    }
  }

  for (let mbX = 0; mbX < mbW; mbX++) {
    const block = mbData[mbX]

    // Rotate left-margin samples in from the previously reconstructed block.
    if (mbX > 0) {
      for (let j = -1; j < 16; j++) {
        for (let k = 0; k < 4; k++) yuvB[yDst + j * BPS - 4 + k] = yuvB[yDst + j * BPS + 12 + k]
      }
      for (let j = -1; j < 8; j++) {
        for (let k = 0; k < 4; k++) {
          yuvB[uDst + j * BPS - 4 + k] = yuvB[uDst + j * BPS + 4 + k]
          yuvB[vDst + j * BPS - 4 + k] = yuvB[vDst + j * BPS + 4 + k]
        }
      }
    }

    // Load top-row samples from yuvT.
    if (mbY > 0) {
      for (let i = 0; i < 16; i++) yuvB[yDst - BPS + i] = yuvT.y[mbX * 16 + i]
      for (let i = 0; i < 8; i++) {
        yuvB[uDst - BPS + i] = yuvT.u[mbX * 8 + i]
        yuvB[vDst - BPS + i] = yuvT.v[mbX * 8 + i]
      }
    }

    // Predict + add residuals.
    if (block.isI4x4) {
      // Set up the "above-right" extension for VE/LD/VL predictors.
      if (mbY > 0) {
        if (mbX >= mbW - 1) {
          // Right-edge MB: replicate the top-right pixel.
          for (let i = 0; i < 4; i++) yuvB[yDst - BPS + 16 + i] = yuvT.y[mbX * 16 + 15]
        }
        else {
          for (let i = 0; i < 4; i++) yuvB[yDst - BPS + 16 + i] = yuvT.y[(mbX + 1) * 16 + i]
        }
      }
      // Replicate the top-right 4 bytes at rows 3, 7, 11 (the rows above
      // sub-blocks 7, 11, 15) so their VE4/LD4/VL4 predictors can read
      // top[4..7] from those positions. Mirrors libwebp's
      // `top_right[BPS] = top_right[2*BPS] = top_right[3*BPS] = top_right[0]`,
      // where BPS is interpreted as a uint32_t-unit stride.
      for (let r = 0; r < 4; r++) {
        const src = yDst - BPS + 16 + r
        for (let dstRow = 1; dstRow < 4; dstRow++) {
          yuvB[yDst + dstRow * 4 * BPS - BPS + 16 + r] = yuvB[src]
        }
      }

      let bits = block.nonZeroY >>> 0
      for (let n = 0; n < 16; n++) {
        const off = yDst + K_SCAN[n]
        // 4×4 B-modes don't get edge-variant substitution — predictors
        // read directly from the dst buffer's surrounding context, which
        // ReconstructRow has pre-initialised with 127/129 sentinels.
        predictLuma4(block.imodes[n], yuvB, off)
        doTransform(bits, block.coeffs, n * 16, yuvB, off)
        bits = (bits << 2) >>> 0
      }
    }
    else {
      const pred = checkMode(mbX, mbY, block.imodes[0])
      predictLuma16(pred, yuvB, yDst)
      let bits = block.nonZeroY >>> 0
      if (bits !== 0) {
        for (let n = 0; n < 16; n++) {
          doTransform(bits, block.coeffs, n * 16, yuvB, yDst + K_SCAN[n])
          bits = (bits << 2) >>> 0
        }
      }
    }

    // Chroma.
    {
      const bitsUV = block.nonZeroUV >>> 0
      const pred = checkMode(mbX, mbY, block.uvMode)
      predictChroma8(pred, yuvB, uDst)
      predictChroma8(pred, yuvB, vDst)
      doUVTransform(bitsUV >>> 0, block.coeffs, 16 * 16, yuvB, uDst)
      doUVTransform((bitsUV >>> 8) >>> 0, block.coeffs, 20 * 16, yuvB, vDst)
    }

    // Stash bottom row into yuvT for the next MB row.
    if (mbY < mbH - 1) {
      for (let i = 0; i < 16; i++) yuvT.y[mbX * 16 + i] = yuvB[yDst + 15 * BPS + i]
      for (let i = 0; i < 8; i++) {
        yuvT.u[mbX * 8 + i] = yuvB[uDst + 7 * BPS + i]
        yuvT.v[mbX * 8 + i] = yuvB[vDst + 7 * BPS + i]
      }
    }

    // Copy reconstructed MB into the frame cache.
    for (let j = 0; j < 16; j++) {
      const c = mbY * 16 * cacheYStride + mbX * 16 + j * cacheYStride
      for (let i = 0; i < 16; i++) cacheY[c + i] = yuvB[yDst + j * BPS + i]
    }
    for (let j = 0; j < 8; j++) {
      const c = mbY * 8 * cacheUVStride + mbX * 8 + j * cacheUVStride
      for (let i = 0; i < 8; i++) {
        cacheU[c + i] = yuvB[uDst + j * BPS + i]
        cacheV[c + i] = yuvB[vDst + j * BPS + i]
      }
    }
  }
}

/** libwebp's CheckMode: substitute DC variants when neighbours are missing. */
function checkMode(mbX: number, mbY: number, mode: number): number {
  if (mode === DC_PRED) {
    if (mbX === 0) return mbY === 0 ? B_DC_PRED_NOTOPLEFT : B_DC_PRED_NOLEFT
    return mbY === 0 ? B_DC_PRED_NOTOP : DC_PRED
  }
  return mode
}

/** doTransform — pick a transform variant from the 2 high bits of `bits`. */
function doTransform(
  bits: number, coeffs: Int16Array, coeffOff: number,
  dst: Uint8Array, dstOff: number,
): void {
  const code = (bits >>> 30) & 3
  switch (code) {
    case 3:
      idct4x4Add(coeffs.subarray(coeffOff, coeffOff + 16) as Int16Array, dst, BPS, dstOff)
      break
    case 2:
      // libwebp's TransformAC3 (only [0], [1], [4] are non-zero).
      // For correctness we just call the full IDCT — it produces the same
      // result and AC3 is just a perf optimisation.
      idct4x4Add(coeffs.subarray(coeffOff, coeffOff + 16) as Int16Array, dst, BPS, dstOff)
      break
    case 1:
      transformDC(coeffs.subarray(coeffOff, coeffOff + 16) as Int16Array, dst, BPS, dstOff)
      break
    default:
      break
  }
}

/** doUVTransform — chroma path; uses TransformDC fast-path when only DCs are non-zero. */
function doUVTransform(
  bits: number, coeffs: Int16Array, coeffOff: number,
  dst: Uint8Array, dstOff: number,
): void {
  if ((bits & 0xFF) === 0) return
  if ((bits & 0xAA) !== 0) {
    // Some AC is non-zero — full IDCT for all 4 sub-blocks.
    idct4x4Add(coeffs.subarray(coeffOff + 0 * 16, coeffOff + 1 * 16) as Int16Array, dst, BPS, dstOff)
    idct4x4Add(coeffs.subarray(coeffOff + 1 * 16, coeffOff + 2 * 16) as Int16Array, dst, BPS, dstOff + 4)
    idct4x4Add(coeffs.subarray(coeffOff + 2 * 16, coeffOff + 3 * 16) as Int16Array, dst, BPS, dstOff + 4 * BPS)
    idct4x4Add(coeffs.subarray(coeffOff + 3 * 16, coeffOff + 4 * 16) as Int16Array, dst, BPS, dstOff + 4 * BPS + 4)
  }
  else {
    // DC-only fast path per sub-block.
    if (coeffs[coeffOff + 0 * 16]) transformDC(coeffs.subarray(coeffOff + 0 * 16, coeffOff + 1 * 16) as Int16Array, dst, BPS, dstOff)
    if (coeffs[coeffOff + 1 * 16]) transformDC(coeffs.subarray(coeffOff + 1 * 16, coeffOff + 2 * 16) as Int16Array, dst, BPS, dstOff + 4)
    if (coeffs[coeffOff + 2 * 16]) transformDC(coeffs.subarray(coeffOff + 2 * 16, coeffOff + 3 * 16) as Int16Array, dst, BPS, dstOff + 4 * BPS)
    if (coeffs[coeffOff + 3 * 16]) transformDC(coeffs.subarray(coeffOff + 3 * 16, coeffOff + 4 * 16) as Int16Array, dst, BPS, dstOff + 4 * BPS + 4)
  }
}

// ---------------------------------------------------------------------------
// Per-row loop filter (port of libwebp's FilterRow)
// ---------------------------------------------------------------------------

function filterRow(
  mbY: number, mbW: number, mbH: number, mbData: MBData[],
  fStrengths: Array<[ReturnType<typeof precomputeFilterStrength>, ReturnType<typeof precomputeFilterStrength>]>,
  filterType: number,
  cacheY: Uint8Array, cacheU: Uint8Array, cacheV: Uint8Array,
  cacheYStride: number, cacheUVStride: number,
): void {
  void mbH
  for (let mbX = 0; mbX < mbW; mbX++) {
    const block = mbData[mbX]
    const f = fStrengths[block.segment][block.isI4x4 ? 1 : 0]
    if (f.limit === 0) continue
    const inner = f.inner || (block.nonZeroY !== 0 || block.nonZeroUV !== 0)
    const yOff = mbY * 16 * cacheYStride + mbX * 16
    const limit = f.limit + 4 // outer (MB-edge) limit
    const innerLimit = f.limit
    if (filterType === 1) {
      if (mbX > 0) simpleHFilter16(cacheY, yOff, cacheYStride, limit)
      if (inner) simpleHFilter16i(cacheY, yOff, cacheYStride, innerLimit)
      if (mbY > 0) simpleVFilter16(cacheY, yOff, cacheYStride, limit)
      if (inner) simpleVFilter16i(cacheY, yOff, cacheYStride, innerLimit)
    }
    else {
      const uOff = mbY * 8 * cacheUVStride + mbX * 8
      const vOff = mbY * 8 * cacheUVStride + mbX * 8
      const hev = f.hevThresh
      const il = f.iLevel
      if (mbX > 0) {
        hFilter16(cacheY, yOff, cacheYStride, limit, il, hev)
        hFilter8(cacheU, uOff, cacheV, vOff, cacheUVStride, limit, il, hev)
      }
      if (inner) {
        hFilter16i(cacheY, yOff, cacheYStride, innerLimit, il, hev)
        hFilter8i(cacheU, uOff, cacheV, vOff, cacheUVStride, innerLimit, il, hev)
      }
      if (mbY > 0) {
        vFilter16(cacheY, yOff, cacheYStride, limit, il, hev)
        vFilter8(cacheU, uOff, cacheV, vOff, cacheUVStride, limit, il, hev)
      }
      if (inner) {
        vFilter16i(cacheY, yOff, cacheYStride, innerLimit, il, hev)
        vFilter8i(cacheU, uOff, cacheV, vOff, cacheUVStride, innerLimit, il, hev)
      }
    }
  }
}

// ---------------------------------------------------------------------------
// YUV 4:2:0 → RGBA — port of libwebp's `VP8YuvToRgb` (src/dsp/yuv.h).
// ---------------------------------------------------------------------------

const YUV_FIX2 = 6
const YUV_MASK2 = (256 << YUV_FIX2) - 1

function multHi(v: number, coeff: number): number {
  return (v * coeff) >> 8
}

function vp8Clip8(v: number): number {
  return ((v & ~YUV_MASK2) === 0) ? (v >> YUV_FIX2) : (v < 0) ? 0 : 255
}

function vp8YUVToR(y: number, v: number): number {
  return vp8Clip8(multHi(y, 19077) + multHi(v, 26149) - 14234)
}
function vp8YUVToG(y: number, u: number, v: number): number {
  return vp8Clip8(multHi(y, 19077) - multHi(u, 6419) - multHi(v, 13320) + 8708)
}
function vp8YUVToB(y: number, u: number): number {
  return vp8Clip8(multHi(y, 19077) + multHi(u, 33050) - 17685)
}

function yuv420ToRgba(
  Y: Uint8Array, U: Uint8Array, V: Uint8Array,
  yStride: number, uvStride: number,
  W: number, H: number, out: Uint8Array,
): void {
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const yi = Y[y * yStride + x]
      const ui = U[(y >> 1) * uvStride + (x >> 1)]
      const vi = V[(y >> 1) * uvStride + (x >> 1)]
      const o = (y * W + x) * 4
      out[o] = vp8YUVToR(yi, vi)
      out[o + 1] = vp8YUVToG(yi, ui, vi)
      out[o + 2] = vp8YUVToB(yi, ui)
      out[o + 3] = 255
    }
  }
}

// Suppress unused-import warnings for symbols re-exported through other modules.
void B_PRED
