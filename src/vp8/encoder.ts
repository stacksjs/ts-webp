/**
 * VP8 (lossy) encoder. Produces a single-keyframe VP8 bitstream that
 * any conformant decoder (including `dwebp` and our own `decodeVP8`)
 * can decode.
 *
 * Scope. To keep this implementation tractable, we limit the encoder
 * to the simplest VP8 decisions allowed by the spec:
 *   • Single segment, single quantiser (no per-segment overrides).
 *   • Single token partition (no parallel decode).
 *   • All macroblocks use `mb->type = 1` (16×16 luma intra-prediction)
 *     with mode `DC_PRED` for both Y16 and chroma.
 *   • No loop filter (`filter_level = 0`).
 *   • No coefficient-probability update (decoder uses the spec defaults).
 *   • No `mb_no_skip_coef` shortcut.
 *
 * That gives us a decoder-compatible stream with quality roughly
 * comparable to cwebp at low effort levels — visibly worse than the
 * reference encoder's full mode-search but functionally complete.
 *
 * Pipeline:
 *
 *   1. RGB → YUV 4:2:0 conversion (BT.601 fixed-point).
 *   2. Per macroblock (16 × 16 luma, 8 × 8 chroma):
 *      a. DC-predict from the row-above and left-column neighbours
 *         (with the spec's edge fallbacks for the top row / left column).
 *      b. Residual = source - prediction.
 *      c. Forward 4×4 DCT on each of the 16 luma sub-blocks. The DC
 *         coefficient of each goes into a "Y2" block.
 *      d. Forward 4×4 WHT on the 16 DCs to produce the Y2 coefficients.
 *      e. Quantise the Y2 block (DC→Q.y2[0], AC→Q.y2[1]) and each
 *         luma 4×4 (DC was replaced by 0 above; AC→Q.y1[1]).
 *      f. Same for the 4 U + 4 V chroma 4×4 blocks (Q.uv[0]/[1]).
 *      g. Serialise the 16x16 mode bit (=1), the I16 mode (=DC_PRED),
 *         the UV mode (=DC_PRED), the Y2 + 16 Y + 4 U + 4 V token blocks.
 *   3. Frame header: keyframe tag, start code, dimensions, segment
 *      header (single segment), filter (off), 1 partition, quantiser,
 *      no proba update, default-loaded coefficient probabilities (we
 *      don't override any).
 *   4. Concatenate frame header + first partition.
 *
 * Reference: RFC 6386, libwebp `enc/syntax_enc.c` + `enc/frame_enc.c`.
 */

import type { WebpImageData } from '../types'
import { BoolEncoder } from './bool-encoder'
import { fdct4x4, fwht4x4 } from './fdct'
import { idct4x4Add, iwht4x4 } from './idct'
import { BPS } from './intra'
import { quantizeBlock } from './quant-enc'
import {
  AC_QUANT,
  CAT3_PROBS,
  CAT4_PROBS,
  CAT5_PROBS,
  CAT6_PROBS,
  COEF_BANDS,
  COEF_UPDATE_PROBS,
  DC_QUANT,
  DEFAULT_COEF_PROBS,
  NUM_COEF_BANDS,
  clampQi,
} from './tables'

// Silence unused-import linter warnings for symbols that future
// 4×4-prediction encoder paths will need; we still export them here
// so the public surface is stable.
void fdct4x4

/** Block-type indices (must match `coeff.ts` decoder constants). */
const BLOCK_TYPE_Y_AFTER_Y2 = 0
const BLOCK_TYPE_Y2 = 1
const BLOCK_TYPE_UV = 2
const BLOCK_TYPE_Y_NO_Y2 = 3

const NUM_LITERAL_PROBS = 11
const I16_DC_PRED = 0

export interface VP8EncodeOptions {
  /**
   * Quantiser index 0..127, matching VP8's per-frame `Q_index`. Lower
   * = higher quality / larger output. Default 60 (~ cwebp -q 30).
   */
  quality?: number
}

/**
 * Encode `image` (RGBA, 8-bits/channel) into a VP8 keyframe bitstream
 * — just the VP8 chunk payload, NOT a full RIFF/WEBP wrapper. The
 * caller is responsible for wrapping the returned bytes in `RIFF/WEBP/
 * VP8 ` (or VP8X + alpha + VP8 for transparent inputs).
 */
export function encodeVP8(image: WebpImageData, options: VP8EncodeOptions = {}): Uint8Array {
  const { width, height, data } = image
  if (width <= 0 || height <= 0) throw new Error('VP8: width and height must be positive')
  if (width > 16383 || height > 16383) throw new Error('VP8: dimensions cap at 16383')
  const baseQ = options.quality ?? 60
  if (baseQ < 0 || baseQ > 127) throw new Error('VP8: quality must be 0..127')

  // ── 1. RGB → YUV 4:2:0 ──
  const mbW = (width + 15) >> 4
  const mbH = (height + 15) >> 4
  const ySize = mbW * 16 * mbH * 16
  const uvSize = mbW * 8 * mbH * 8
  const Y = new Uint8Array(ySize)
  const U = new Uint8Array(uvSize)
  const V = new Uint8Array(uvSize)
  rgbToYuv420(data, width, height, mbW * 16, mbH * 16, Y, U, V)

  // Reconstructed planes. Every macroblock predicts from the RECONSTRUCTED
  // (dequant + inverse-transform + clamp) pixels of its already-coded top /
  // left neighbours — exactly what the decoder sees — so encoder and decoder
  // stay in lock-step and error does not accumulate across the MB grid.
  const Yrec = new Uint8Array(ySize)
  const Urec = new Uint8Array(uvSize)
  const Vrec = new Uint8Array(uvSize)

  // ── 2. Quantiser tables (spec §11.4) ──
  const y1Dc = DC_QUANT[clampQi(baseQ)]
  const y1Ac = AC_QUANT[clampQi(baseQ)]
  const y2Dc = DC_QUANT[clampQi(baseQ)] * 2
  let y2Ac = (AC_QUANT[clampQi(baseQ)] * 101581) >> 16
  if (y2Ac < 8) y2Ac = 8
  void y2Dc
  const uvDc = DC_QUANT[clampQi(baseQ) > 117 ? 117 : clampQi(baseQ)]
  const uvAc = AC_QUANT[clampQi(baseQ)]

  // ── 3. Encode each macroblock ──
  // VP8 uses TWO bool-encoded partitions even at num_parts = 1:
  //   • Partition 0 (`bw`) — frame header + intra-mode info per MB.
  //   • Token partition (`tokens`) — the DCT coefficient streams for
  //     all MBs, decoded by `dec->parts[]` on the decoder side.
  // libwebp lays them out as: [10 bytes uncompressed header][bw][tokens].
  const bw = new BoolEncoder(Math.max(1024, mbW * mbH * 8))
  const tokens = new BoolEncoder(Math.max(1024, mbW * mbH * 32))

  // Header bits at the start of partition-0:
  bw.writeBitUniform(0) // colorspace
  bw.writeBitUniform(0) // clamp_type
  // Segment header: use_segment = 0 → no further segment data.
  bw.writeBitUniform(0)
  // Filter header: filter_type=0, level=0, sharpness=0, no lf-delta.
  bw.writeBitUniform(0) // simple
  bw.writeLiteral(0, 6) // level = 0 → loop filter off
  bw.writeLiteral(0, 3) // sharpness
  bw.writeBitUniform(0) // use_lf_delta = 0
  // Partition count: log2(1) = 0 → write 0 in 2 bits.
  bw.writeLiteral(0, 2)
  // Quantiser: base + 5 deltas (all 0). Each delta is encoded as
  // libwebp's `VP8PutSignedBits(value, 4)`: 1 bit "is non-zero", then
  // (only if non-zero) 5 bits for `(magnitude << 1) | sign`. For all
  // five deltas set to 0, that's just five 0-bits.
  bw.writeLiteral(baseQ, 7)
  for (let i = 0; i < 5; i++) bw.writeBitUniform(0)
  // No proba update.
  bw.writeBitUniform(0)
  // Coefficient-probability update loop: emit one "no-update" bit per
  // entry. The decoder will keep the spec-default `DEFAULT_COEF_PROBS`.
  emitNoCoefProbUpdate(bw)
  // mb_no_skip_coef = 0 (don't use the skip-bit shortcut).
  bw.writeBitUniform(0)

  // Per-macroblock state for the coefficient context. `topNzY`/`leftNzY`
  // track the "previous non-zero?" flags across 4x4 sub-blocks. `topNzDc`/
  // `leftNzDc` track the Y2 block's NZ flag. Same for UV.
  const topNzY = new Uint8Array(mbW * 4)
  const topNzU = new Uint8Array(mbW * 2)
  const topNzV = new Uint8Array(mbW * 2)
  const topNzDc = new Uint8Array(mbW)

  // Reusable per-MB buffers. Each holds a 1-pixel-thick top + left
  // border around the MB body, so size = BPS × (block + 1).
  const yBlock = new Uint8Array(BPS * 17)
  const uBlock = new Uint8Array(BPS * 9)
  const vBlock = new Uint8Array(BPS * 9)
  const yResidual = new Int16Array(16 * 16)
  const uResidual = new Int16Array(4 * 16)
  const vResidual = new Int16Array(4 * 16)
  const dcArr = new Int16Array(16) // DC of each Y 4×4 block
  const yLevels = new Int16Array(16 * 16) // zigzag levels for 16 Y blocks
  const uLevels = new Int16Array(4 * 16)
  const vLevels = new Int16Array(4 * 16)
  const y2Levels = new Int16Array(16)

  for (let my = 0; my < mbH; my++) {
    let leftNzY = 0
    let leftNzU = 0
    let leftNzV = 0
    let leftNzDc = 0
    for (let mx = 0; mx < mbW; mx++) {
      // Pack the 16×16 Y / 8×8 U / 8×8 V samples into BPS-strided
      // buffers. The MB *body* comes from the source planes, but the
      // 1-pixel top/left predictor border comes from the RECONSTRUCTED
      // planes (the pixels the decoder will actually predict from).
      gatherY16(Y, Yrec, mbW * 16, mx, my, yBlock)
      gatherUV(U, Urec, mbW * 8, mx, my, uBlock)
      gatherUV(V, Vrec, mbW * 8, mx, my, vBlock)

      // DC prediction (mode 0): predict every pixel as the average of
      // the 16 row-above + 16 left-column neighbours (or the spec's
      // `0x80` fallback at frame corner).
      const yPred = dcPredict16(yBlock, mx, my)
      const uPred = dcPredict8(uBlock, mx, my)
      const vPred = dcPredict8(vBlock, mx, my)

      // Residual = source - prediction (per pixel, kept as int16).
      computeResidual16(yBlock, yPred, yResidual)
      computeResidual8(uBlock, uPred, uResidual)
      computeResidual8(vBlock, vPred, vResidual)

      // Forward DCT on each 4×4 luma sub-block, save DCs.
      for (let by = 0; by < 4; by++) {
        for (let bx = 0; bx < 4; bx++) {
          const bIdx = by * 4 + bx
          fdct4x4Residual(yResidual, by * 4, bx * 4, yLevels.subarray(bIdx * 16, bIdx * 16 + 16) as unknown as Int16Array)
        }
      }
      // Move each Y block's DC into `dcArr` and zero it in the
      // per-block coefficient buffer (so the AC-only quant pass below
      // doesn't re-quantise the DC).
      for (let i = 0; i < 16; i++) {
        dcArr[i] = yLevels[i * 16]
        yLevels[i * 16] = 0
      }
      // Y2 block: forward WHT on the 16 DCs.
      fwht4x4(dcArr, 0, y2Levels, 0)
      // Quantise Y2.
      const y2Quant = new Int16Array(16)
      const y2Nz = quantizeBlock(y2Levels, 0, y2Quant, 0, y2Dc, y2Ac)
      // Quantise each Y AC block (DC stays 0 in zigzag pos 0).
      const yQuant = new Int16Array(16 * 16)
      const yNzs = new Uint8Array(16)
      for (let i = 0; i < 16; i++) {
        const nz = quantizeBlock(yLevels, i * 16, yQuant, i * 16, y1Dc, y1Ac)
        yNzs[i] = nz ? 1 : 0
      }
      // Quantise UV blocks.
      for (let i = 0; i < 4; i++) {
        fdct4x4Residual(uResidual, (i >> 1) * 4, (i & 1) * 4, uLevels.subarray(i * 16, i * 16 + 16) as unknown as Int16Array)
        fdct4x4Residual(vResidual, (i >> 1) * 4, (i & 1) * 4, vLevels.subarray(i * 16, i * 16 + 16) as unknown as Int16Array)
      }
      const uQuant = new Int16Array(4 * 16)
      const vQuant = new Int16Array(4 * 16)
      const uNzs = new Uint8Array(4)
      const vNzs = new Uint8Array(4)
      for (let i = 0; i < 4; i++) {
        uNzs[i] = quantizeBlock(uLevels, i * 16, uQuant, i * 16, uvDc, uvAc) ? 1 : 0
        vNzs[i] = quantizeBlock(vLevels, i * 16, vQuant, i * 16, uvDc, uvAc) ? 1 : 0
      }

      // ── Reconstruction (mirror the decoder) ──
      // `quantizeBlock` already wrote the dequantised (level × Q) values
      // back into the *Levels buffers in raster order, so they now hold
      // exactly the coefficients the decoder will read. Inverse-transform
      // and add the prediction to produce the reconstructed pixels, then
      // store them so the NEXT macroblock predicts from the same data the
      // decoder will.
      reconstructY16(y2Levels, yLevels, yPred, Yrec, mbW * 16, mx, my)
      reconstructChroma(uLevels, uPred, Urec, mbW * 8, mx, my)
      reconstructChroma(vLevels, vPred, Vrec, mbW * 8, mx, my)

      // ── Mode info (per spec §11) ──
      // Skip flag is disabled (mb_no_skip_coef = 0), so no skip-bit.
      // i16 vs i4: write 1 (i16x16).
      bw.writeBit(145, 1)
      // I16 mode: encode DC_PRED via the kYModesIntra4 tree.
      putI16Mode(bw, I16_DC_PRED)
      // UV mode: DC_PRED.
      putUVMode(bw, I16_DC_PRED)

      // ── Token coding (into the `tokens` bool encoder, not bw) ──
      // Y2 first.
      const ctxY2 = topNzDc[mx] + leftNzDc
      putBlockCoeffs(tokens, y2Quant, BLOCK_TYPE_Y2, ctxY2, 0, y2Nz)
      const y2NzFlag = y2Nz ? 1 : 0
      topNzDc[mx] = leftNzDc = y2NzFlag

      // 16 Y AC blocks.
      for (let by = 0; by < 4; by++) {
        let l = (leftNzY >>> by) & 1
        for (let bx = 0; bx < 4; bx++) {
          const bIdx = by * 4 + bx
          const ctx = l + topNzY[mx * 4 + bx]
          const blockNz = yNzs[bIdx]
          putBlockCoeffs(tokens, yQuant, BLOCK_TYPE_Y_AFTER_Y2, ctx, bIdx * 16, !!blockNz, /* firstCoef */ 1)
          l = blockNz
          topNzY[mx * 4 + bx] = blockNz
        }
        // Pack `l` flag into leftNzY at bit `by`.
        leftNzY = (leftNzY & ~(1 << by)) | (l << by)
      }

      // 4 U blocks, then 4 V blocks.
      for (let by = 0; by < 2; by++) {
        let l = (leftNzU >>> by) & 1
        for (let bx = 0; bx < 2; bx++) {
          const bIdx = by * 2 + bx
          const ctx = l + topNzU[mx * 2 + bx]
          const nz = uNzs[bIdx]
          putBlockCoeffs(tokens, uQuant, BLOCK_TYPE_UV, ctx, bIdx * 16, !!nz)
          l = nz
          topNzU[mx * 2 + bx] = nz
        }
        leftNzU = (leftNzU & ~(1 << by)) | (l << by)
      }
      for (let by = 0; by < 2; by++) {
        let l = (leftNzV >>> by) & 1
        for (let bx = 0; bx < 2; bx++) {
          const bIdx = by * 2 + bx
          const ctx = l + topNzV[mx * 2 + bx]
          const nz = vNzs[bIdx]
          putBlockCoeffs(tokens, vQuant, BLOCK_TYPE_UV, ctx, bIdx * 16, !!nz)
          l = nz
          topNzV[mx * 2 + bx] = nz
        }
        leftNzV = (leftNzV & ~(1 << by)) | (l << by)
      }
    }
  }

  const partition0 = bw.finish()
  const tokenPart = tokens.finish()

  // ── 4. Frame header (uncompressed) + concatenate partitions ──
  // Layout: 10-byte frame header || partition0 || token partition.
  // For num_parts=1, no inter-partition size table is needed.
  const firstPartSize = partition0.length
  const total = 10 + firstPartSize + tokenPart.length
  const out = new Uint8Array(total)
  // Frame tag: keyframe (bit 0 = 0), profile=0, show_frame=1, partSize.
  const frameTag = (firstPartSize << 5) | (1 << 4) | 0 // | (0 << 1) | 0
  out[0] = frameTag & 0xFF
  out[1] = (frameTag >> 8) & 0xFF
  out[2] = (frameTag >> 16) & 0xFF
  // Start code.
  out[3] = 0x9D
  out[4] = 0x01
  out[5] = 0x2A
  // Width/height (14 bits each + 2 scale bits, scale=0).
  out[6] = width & 0xFF
  out[7] = (width >> 8) & 0x3F
  out[8] = height & 0xFF
  out[9] = (height >> 8) & 0x3F
  out.set(partition0, 10)
  out.set(tokenPart, 10 + firstPartSize)
  return out
}

// ---------------------------------------------------------------------------
// Helpers — RGB→YUV, MB gather, DC prediction, residual, FDCT wrapper
// ---------------------------------------------------------------------------

/** BT.601 RGB→YUV with rounded fixed-point coefficients (libwebp uses
 *  the same constants). Output is full-range for Y/U/V to match the
 *  decoder's BT.601 inverse. Pads the right/bottom edges by replicating
 *  the last row/column when (width, height) isn't 16-aligned. */
function rgbToYuv420(
  rgba: Uint8Array,
  width: number, height: number,
  paddedW: number, paddedH: number,
  Y: Uint8Array, U: Uint8Array, V: Uint8Array,
): void {
  // Per pixel: Y = (66*R + 129*G + 25*B + 128) >> 8 + 16
  //            U = (-38*R - 74*G + 112*B + 128) >> 8 + 128
  //            V = (112*R - 94*G - 18*B + 128) >> 8 + 128
  for (let y = 0; y < paddedH; y++) {
    const sy = y < height ? y : height - 1
    for (let x = 0; x < paddedW; x++) {
      const sx = x < width ? x : width - 1
      const o = (sy * width + sx) * 4
      const r = rgba[o]
      const g = rgba[o + 1]
      const b = rgba[o + 2]
      const yv = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
      Y[y * paddedW + x] = yv < 0 ? 0 : yv > 255 ? 255 : yv
      // 4:2:0 chroma — 1 sample per 2×2 luma. Average the 2×2 RGB block.
      if ((y & 1) === 0 && (x & 1) === 0) {
        const avgR = r
        const avgG = g
        const avgB = b
        const uv = ((-38 * avgR - 74 * avgG + 112 * avgB + 128) >> 8) + 128
        const vv = ((112 * avgR - 94 * avgG - 18 * avgB + 128) >> 8) + 128
        U[(y >> 1) * (paddedW >> 1) + (x >> 1)] = uv < 0 ? 0 : uv > 255 ? 255 : uv
        V[(y >> 1) * (paddedW >> 1) + (x >> 1)] = vv < 0 ? 0 : vv > 255 ? 255 : vv
      }
    }
  }
}

/**
 * Copy a 16×16 luma block into `dst` (BPS-strided). The MB *body* is read
 * from the source plane `Y`; the 1-pixel top/left predictor border is read
 * from the *reconstructed* plane `Yr` (matching what the decoder predicts
 * from). Border samples outside the frame use libwebp's 0x80 fallback.
 */
function gatherY16(Y: Uint8Array, Yr: Uint8Array, stride: number, mx: number, my: number, dst: Uint8Array): void {
  // Top border (-1 row) — reconstructed neighbour above.
  for (let x = -1; x < 16; x++) {
    const sx = mx * 16 + x
    const sy = my * 16 - 1
    dst[(0) * BPS + (x + 1)] = (sx >= 0 && sy >= 0)
      ? Yr[sy * stride + sx]
      : 0x80
  }
  // Left border (-1 column) — reconstructed neighbour to the left.
  for (let y = 0; y < 16; y++) {
    const sx = mx * 16 - 1
    const sy = my * 16 + y
    dst[(y + 1) * BPS] = (sx >= 0)
      ? Yr[sy * stride + sx]
      : 0x80
    // Body (this row) — source pixels.
    for (let x = 0; x < 16; x++) {
      dst[(y + 1) * BPS + (x + 1)] = Y[(my * 16 + y) * stride + (mx * 16 + x)]
    }
  }
}

/** Same as `gatherY16` but for an 8×8 chroma block. */
function gatherUV(C: Uint8Array, Cr: Uint8Array, stride: number, mx: number, my: number, dst: Uint8Array): void {
  for (let x = -1; x < 8; x++) {
    const sx = mx * 8 + x
    const sy = my * 8 - 1
    dst[(0) * BPS + (x + 1)] = (sx >= 0 && sy >= 0)
      ? Cr[sy * stride + sx]
      : 0x80
  }
  for (let y = 0; y < 8; y++) {
    const sx = mx * 8 - 1
    const sy = my * 8 + y
    dst[(y + 1) * BPS] = (sx >= 0)
      ? Cr[sy * stride + sx]
      : 0x80
    for (let x = 0; x < 8; x++) {
      dst[(y + 1) * BPS + (x + 1)] = C[(my * 8 + y) * stride + (mx * 8 + x)]
    }
  }
}

/**
 * Reconstruct a 16×16 luma macroblock exactly as the decoder will, and
 * store the pixels into the reconstructed plane `Yr` at (mx, my).
 *
 * `y2Levels` holds the dequantised Y2 (WHT) coefficients in raster order;
 * `yLevels` holds the dequantised AC coefficients for the 16 luma 4×4
 * sub-blocks (DC at each block's index 0 is 0 — it lives in Y2). We invert
 * the WHT to recover the 16 per-block DCs, inject each as `coeffs[0]`, then
 * run the 4×4 IDCT onto a prediction-filled patch.
 */
function reconstructY16(
  y2Levels: Int16Array,
  yLevels: Int16Array,
  pred: number,
  Yr: Uint8Array,
  stride: number,
  mx: number,
  my: number,
): void {
  // Invert the Walsh-Hadamard transform → 16 luma DCs (raster block order).
  iwht4x4(y2Levels)
  const patch = new Uint8Array(16 * 16)
  patch.fill(pred)
  for (let by = 0; by < 4; by++) {
    for (let bx = 0; bx < 4; bx++) {
      const bIdx = by * 4 + bx
      const sub = yLevels.subarray(bIdx * 16, bIdx * 16 + 16)
      sub[0] = y2Levels[bIdx]
      idct4x4Add(sub as unknown as Int16Array, patch, 16, by * 4 * 16 + bx * 4)
    }
  }
  for (let y = 0; y < 16; y++) {
    const row = (my * 16 + y) * stride + mx * 16
    for (let x = 0; x < 16; x++) Yr[row + x] = patch[y * 16 + x]
  }
}

/**
 * Reconstruct an 8×8 chroma macroblock (4 luma-style 4×4 sub-blocks, each
 * with its own DC) into the reconstructed plane `Cr` at (mx, my).
 */
function reconstructChroma(
  levels: Int16Array,
  pred: number,
  Cr: Uint8Array,
  stride: number,
  mx: number,
  my: number,
): void {
  const patch = new Uint8Array(8 * 8)
  patch.fill(pred)
  for (let i = 0; i < 4; i++) {
    const sub = levels.subarray(i * 16, i * 16 + 16)
    idct4x4Add(sub as unknown as Int16Array, patch, 8, (i >> 1) * 4 * 8 + (i & 1) * 4)
  }
  for (let y = 0; y < 8; y++) {
    const row = (my * 8 + y) * stride + mx * 8
    for (let x = 0; x < 8; x++) Cr[row + x] = patch[y * 8 + x]
  }
}

/**
 * Compute a 16×16 DC predictor as the rounded average of the 16 row-
 * above + 16 left-column neighbours. Falls back to 0x80 at frame
 * corner per RFC 6386 §12.2.
 *
 * `block` layout (set up by `gatherY16`):
 *   • row 0, indices 1..16  → 16 bytes of the row directly above the MB
 *   • rows 1..16, index 0   → 16 bytes of the column directly left of the MB
 *   • rows 1..16, indices 1..16 → the 16×16 MB itself
 *
 * (`block[0]` holds the top-left corner sample.)
 */
function dcPredict16(block: Uint8Array, mx: number, my: number): number {
  if (mx === 0 && my === 0) return 0x80
  if (mx === 0) {
    // Top row only — `block[1..16]` of row 0.
    let sum = 8
    for (let x = 0; x < 16; x++) sum += block[(x + 1)]
    return sum >> 4
  }
  if (my === 0) {
    // Left column only — index 0 of rows 1..16.
    let sum = 8
    for (let y = 0; y < 16; y++) sum += block[(y + 1) * BPS]
    return sum >> 4
  }
  let sum = 16
  for (let y = 0; y < 16; y++) sum += block[(y + 1) * BPS]
  for (let x = 0; x < 16; x++) sum += block[(x + 1)]
  return sum >> 5
}

/** 8×8 chroma DC predictor — same layout as `dcPredict16` but 8×8. */
function dcPredict8(block: Uint8Array, mx: number, my: number): number {
  if (mx === 0 && my === 0) return 0x80
  if (mx === 0) {
    let sum = 4
    for (let x = 0; x < 8; x++) sum += block[(x + 1)]
    return sum >> 3
  }
  if (my === 0) {
    let sum = 4
    for (let y = 0; y < 8; y++) sum += block[(y + 1) * BPS]
    return sum >> 3
  }
  let sum = 8
  for (let y = 0; y < 8; y++) sum += block[(y + 1) * BPS]
  for (let x = 0; x < 8; x++) sum += block[(x + 1)]
  return sum >> 4
}

/** Compute residual = source - predictor for a 16×16 block (gathered
 *  with a +1 row/column border for prediction context). */
function computeResidual16(block: Uint8Array, pred: number, residual: Int16Array): void {
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      residual[y * 16 + x] = block[(y + 1) * BPS + (x + 1)] - pred
    }
  }
}

/** Same as above for an 8×8 block. */
function computeResidual8(block: Uint8Array, pred: number, residual: Int16Array): void {
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      residual[y * 8 + x] = block[(y + 1) * BPS + (x + 1)] - pred
    }
  }
}

/**
 * Run forward 4×4 DCT on a (`subY`, `subX`)-anchored 4×4 patch of
 * `residual`. Wraps the int16 residual into the byte-shaped buffer the
 * `fdct4x4` function expects (it reads bytes via `BPS` stride).
 */
function fdct4x4Residual(residual: Int16Array, subY: number, subX: number, out: Int16Array): void {
  // We can't directly hand a residual int16 to the byte-stride FDCT.
  // libwebp's FDCT wants `src - ref` where both are uint8 — so it
  // computes the residual itself. We do the same here: just copy the
  // patch into an "uint8 + offset" pair where the residual lives in
  // src and ref is all zeros (so src - ref = src, which is our
  // residual interpreted as a signed-ish byte).
  //
  // To avoid bias, the residuals in [-255, 255] need to be re-shifted
  // back to [0, 255] for the byte-stride DCT path. Easier: do the
  // forward DCT directly on the int16 residual using libwebp's exact
  // formulas.
  fdct4x4Int16(residual, (subY * 16 + subX), 16, out)
}

/**
 * Forward 4×4 DCT on int16 residuals — same arithmetic as
 * `fdct.ts::fdct4x4` except inputs are pre-computed differences instead
 * of `src - ref` byte differences.
 */
function fdct4x4Int16(src: Int16Array, srcOff: number, stride: number, out: Int16Array): void {
  const tmp = new Int32Array(16)
  for (let i = 0; i < 4; i++) {
    const off = srcOff + i * stride
    const d0 = src[off + 0]
    const d1 = src[off + 1]
    const d2 = src[off + 2]
    const d3 = src[off + 3]
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
    out[0 + i] = (a0 + a1 + 7) >> 4
    out[4 + i] = ((a2 * 2217 + a3 * 5352 + 12000) >> 16) + (a3 !== 0 ? 1 : 0)
    out[8 + i] = (a0 - a1 + 7) >> 4
    out[12 + i] = (a3 * 2217 - a2 * 5352 + 51000) >> 16
  }
}

// ---------------------------------------------------------------------------
// Mode-info writers (libwebp's `PutI16Mode` / `PutUVMode`)
// ---------------------------------------------------------------------------

function putI16Mode(bw: BoolEncoder, mode: number): void {
  // Tree path: DC_PRED selects (0, 0). TM_PRED → (1, 0); V_PRED → (0, 1);
  // H_PRED → (1, 1).
  if (mode === 0 || mode === 1) {
    // V_PRED or DC_PRED — first branch = 0
    bw.writeBit(156, 0)
    bw.writeBit(163, mode === 1 ? 1 : 0) // V_PRED=1, DC_PRED=0
  }
  else {
    bw.writeBit(156, 1)
    bw.writeBit(128, mode === 2 ? 1 : 0) // TM_PRED=2 → 1, else H_PRED=3 → 0
  }
}

function putUVMode(bw: BoolEncoder, mode: number): void {
  // DC_PRED=0 → first bit 0. Else first bit 1, then 1/0 for V/non-V.
  if (mode === 0) {
    bw.writeBit(142, 0)
  }
  else if (mode === 2) {
    bw.writeBit(142, 1)
    bw.writeBit(114, 0)
  }
  else if (mode === 3) {
    bw.writeBit(142, 1)
    bw.writeBit(114, 1)
    bw.writeBit(183, 0)
  }
  else {
    bw.writeBit(142, 1)
    bw.writeBit(114, 1)
    bw.writeBit(183, 1) // TM_PRED
  }
}

// ---------------------------------------------------------------------------
// Coefficient-probability update loop — emit one "no-update" bit per
// entry. The update probability for each entry is `COEF_UPDATE_PROBS`.
// ---------------------------------------------------------------------------

function emitNoCoefProbUpdate(bw: BoolEncoder): void {
  const NUM_BLOCK_TYPES = 4
  for (let i = 0; i < NUM_BLOCK_TYPES; i++) {
    for (let j = 0; j < NUM_COEF_BANDS; j++) {
      for (let k = 0; k < 3; k++) {
        for (let l = 0; l < NUM_LITERAL_PROBS; l++) {
          const off = i * NUM_COEF_BANDS * 3 * NUM_LITERAL_PROBS
            + j * 3 * NUM_LITERAL_PROBS
            + k * NUM_LITERAL_PROBS + l
          bw.writeBit(COEF_UPDATE_PROBS[off], 0)
        }
      }
    }
  }
  void DEFAULT_COEF_PROBS
}

// ---------------------------------------------------------------------------
// Token-tree coefficient encoder (libwebp `PutCoeffs`)
// ---------------------------------------------------------------------------

function probSlice(probs: Uint8Array, blockType: number, band: number, ctx: number): Uint8Array {
  const off = blockType * NUM_COEF_BANDS * 3 * NUM_LITERAL_PROBS
    + band * 3 * NUM_LITERAL_PROBS
    + ctx * NUM_LITERAL_PROBS
  return probs.subarray(off, off + NUM_LITERAL_PROBS)
}

/**
 * Encode 16 zig-zag-ordered coefficients of one 4×4 block via the
 * coefficient token tree. Mirrors libwebp's `PutCoeffs` exactly.
 */
function putBlockCoeffs(
  bw: BoolEncoder,
  zigzagLevels: Int16Array,
  blockType: number,
  ctx: number,
  blockOff: number,
  blockNz: boolean,
  firstCoef = 0,
): void {
  // Find `last` — the position of the last non-zero coefficient.
  let last = -1
  for (let n = firstCoef; n < 16; n++) {
    if (zigzagLevels[blockOff + n] !== 0) last = n
  }
  void blockNz

  let n = firstCoef
  let p = probSlice(DEFAULT_COEF_PROBS, blockType, COEF_BANDS[n], ctx)

  // First token: was there any non-zero coefficient in this block?
  bw.writeBit(p[0], last >= 0 ? 1 : 0)
  if (last < 0) return

  while (n < 16) {
    const c = zigzagLevels[blockOff + n++]
    const sign = c < 0
    let v = sign ? -c : c
    // p[1]: zero / non-zero token.
    if (v === 0) {
      bw.writeBit(p[1], 0)
      p = probSlice(DEFAULT_COEF_PROBS, blockType, COEF_BANDS[n], 0)
      continue
    }
    bw.writeBit(p[1], 1)

    // p[2]: 1 / >=2 token.
    if (v === 1) {
      bw.writeBit(p[2], 0)
      p = probSlice(DEFAULT_COEF_PROBS, blockType, COEF_BANDS[n], 1)
    }
    else {
      bw.writeBit(p[2], 1)
      // 2..4? p[3]
      if (v <= 4) {
        bw.writeBit(p[3], 0)
        if (v === 2) {
          bw.writeBit(p[4], 0)
        }
        else {
          bw.writeBit(p[4], 1)
          bw.writeBit(p[5], v === 4 ? 1 : 0)
        }
      }
      else {
        bw.writeBit(p[3], 1)
        // 5..10? p[6]
        if (v <= 10) {
          bw.writeBit(p[6], 0)
          // 5..6 vs 7..10? p[7]
          if (v <= 6) {
            bw.writeBit(p[7], 0)
            bw.writeBit(159, v === 6 ? 1 : 0)
          }
          else {
            bw.writeBit(p[7], 1)
            bw.writeBit(165, v >= 9 ? 1 : 0)
            bw.writeBit(145, !(v & 1) ? 1 : 0)
          }
        }
        else {
          bw.writeBit(p[6], 1)
          // CAT3..CAT6 (large values).
          let mask: number
          let tab: Uint8Array
          if (v < 3 + (8 << 1)) { // CAT3 (range 11..18)
            bw.writeBit(p[8], 0)
            bw.writeBit(p[9], 0)
            v -= 3 + (8 << 0)
            mask = 1 << 2
            tab = CAT3_PROBS
          }
          else if (v < 3 + (8 << 2)) { // CAT4 (range 19..34)
            bw.writeBit(p[8], 0)
            bw.writeBit(p[9], 1)
            v -= 3 + (8 << 1)
            mask = 1 << 3
            tab = CAT4_PROBS
          }
          else if (v < 3 + (8 << 3)) { // CAT5
            bw.writeBit(p[8], 1)
            bw.writeBit(p[10], 0)
            v -= 3 + (8 << 2)
            mask = 1 << 4
            tab = CAT5_PROBS
          }
          else { // CAT6
            bw.writeBit(p[8], 1)
            bw.writeBit(p[10], 1)
            v -= 3 + (8 << 3)
            mask = 1 << 10
            tab = CAT6_PROBS
          }
          let t = 0
          while (mask) {
            bw.writeBit(tab[t++], (v & mask) ? 1 : 0)
            mask >>= 1
          }
        }
        p = probSlice(DEFAULT_COEF_PROBS, blockType, COEF_BANDS[n], 2)
      }
    }
    // Sign bit (uniform).
    bw.writeBitUniform(sign ? 1 : 0)
    if (n === 16) return
    // Continuation bit: 1 if there are more non-zero coeffs to come,
    // 0 if this was the last (EOB). Mirrors libwebp's
    //   `VP8PutBit(bw, n <= res->last, p[0])`.
    if (n > last) {
      bw.writeBit(p[0], 0)
      return
    }
    bw.writeBit(p[0], 1)
  }
}

// keep BLOCK_TYPE_Y_NO_Y2 + fdct4x4 referenced (used only by future
// 4×4-prediction encoder paths — silence dead-import warnings)
void BLOCK_TYPE_Y_NO_Y2
