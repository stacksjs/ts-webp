/**
 * VP8 keyframe mode-info decoder — 1:1 port of libwebp's
 * `ParseIntraMode` (src/dec/tree_dec.c).
 *
 * For each macroblock we decode:
 *   - segment id (if `segmentation.updateMap` is set)
 *   - skip-coef flag (if `mb_no_skip_coef` is on)
 *   - is_i4x4 (true if Y mode is B_PRED, false for the 16×16 modes)
 *   - if 16×16: yMode ∈ {DC,V,H,TM}; if 4×4: 16 per-block B-modes
 *   - uvMode ∈ {DC,V,H,TM}
 *
 * libwebp keeps two arrays of "B-mode along boundary":
 *   - `intra_t[4*mb_x..4*mb_x+3]` — modes from the bottom row of the MB above
 *   - `intra_l[0..3]` — modes from the right column of the MB to the left
 *
 * For 16×16 MBs both are filled with the chosen yMode. For B_PRED MBs
 * they're updated as we decode each sub-block.
 */
import type { BoolDecoder } from './bool-decoder'
import type { VP8SegmentHeader } from './header'
import {
  B_DC_PRED, B_HD_PRED, B_HE_PRED, B_HU_PRED, B_LD_PRED, B_PRED, B_RD_PRED,
  B_TM_PRED, B_VE_PRED, B_VL_PRED, B_VR_PRED, DC_PRED, H_PRED, TM_PRED, V_PRED,
} from './tables'

export interface MBData {
  /** Segment id (0..3). */
  segment: number
  /** True if the macroblock-level skip-coef flag was set. */
  skip: boolean
  /** True if the macroblock uses B_PRED (4×4 intra prediction). */
  isI4x4: boolean
  /** Y intra mode (when !isI4x4) or 16 per-subblock B-modes (when isI4x4). */
  imodes: Uint8Array
  /** Chroma intra mode. */
  uvMode: number
  /** Coefficient buffer (24 blocks × 16 coefs = 384 entries). */
  coeffs: Int16Array
  /** Bit-packed flags: see libwebp's `non_zero_y` / `non_zero_uv`. */
  nonZeroY: number
  nonZeroUV: number
  /** Whether U/V/Y blocks contributed any non-zero coef (column tracker). */
  nonZeroDc: number
}

/** Decode mode info for one macroblock. Mirrors `ParseIntraMode`. */
export function parseIntraMode(
  br: BoolDecoder,
  segmentation: VP8SegmentHeader,
  useSkipProba: boolean,
  skipP: number,
  topModes: Uint8Array,
  topModesOff: number,
  leftModes: Uint8Array,
  block: MBData,
): void {
  // Segment id (rarely used in WebP).
  if (segmentation.updateMap) {
    const p = segmentation.segmentTreeProbs
    block.segment = !br.readBit(p[0])
      ? br.readBit(p[1])
      : br.readBit(p[2]) + 2
  }
  else {
    block.segment = 0
  }
  if (useSkipProba) {
    block.skip = br.readBit(skipP) === 1
  }

  block.isI4x4 = !br.readBit(145)
  if (!block.isI4x4) {
    // 16×16 prediction (one mode for the whole MB).
    const ymode = br.readBit(156)
      ? (br.readBit(128) ? TM_PRED : H_PRED)
      : (br.readBit(163) ? V_PRED : DC_PRED)
    block.imodes[0] = ymode
    for (let i = 0; i < 4; i++) {
      topModes[topModesOff + i] = ymode
      leftModes[i] = ymode
    }
  }
  else {
    // 4×4 prediction (16 sub-block modes).
    let mIdx = 0
    for (let y = 0; y < 4; y++) {
      let ymode = leftModes[y]
      for (let x = 0; x < 4; x++) {
        const probs = kfBmodeProbsAt(topModes[topModesOff + x], ymode)
        ymode = readBmode(br, probs)
        topModes[topModesOff + x] = ymode
        block.imodes[mIdx++] = ymode
      }
      leftModes[y] = ymode
    }
  }

  // UV mode (DC/V/H/TM).
  block.uvMode = !br.readBit(142)
    ? DC_PRED
    : (!br.readBit(114)
        ? V_PRED
        : (br.readBit(183) ? TM_PRED : H_PRED))
}

/** Decode a 4×4 B-mode given the 9-element `[above, left]` probability vector. */
function readBmode(br: BoolDecoder, prob: Uint8Array): number {
  if (!br.readBit(prob[0])) return B_DC_PRED
  if (!br.readBit(prob[1])) return B_TM_PRED
  if (!br.readBit(prob[2])) return B_VE_PRED
  if (!br.readBit(prob[3])) {
    if (!br.readBit(prob[4])) return B_HE_PRED
    return br.readBit(prob[5]) ? B_VR_PRED : B_RD_PRED
  }
  if (!br.readBit(prob[6])) return B_LD_PRED
  if (!br.readBit(prob[7])) return B_VL_PRED
  return br.readBit(prob[8]) ? B_HU_PRED : B_HD_PRED
}

import { KF_BMODE_PROBS, NUM_BMODES } from './tables'

function kfBmodeProbsAt(above: number, left: number): Uint8Array {
  const off = (above * NUM_BMODES + left) * (NUM_BMODES - 1)
  return KF_BMODE_PROBS.subarray(off, off + (NUM_BMODES - 1))
}
