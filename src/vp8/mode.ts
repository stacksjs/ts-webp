/**
 * VP8 keyframe mode-info decoder.
 *
 * For each macroblock, the encoder records:
 *
 *   - the 16×16 luma prediction mode (one of `DC_PRED`, `V_PRED`,
 *     `H_PRED`, `TM_PRED`, `B_PRED`)
 *   - if that mode is `B_PRED`: the 4×4 prediction mode for each of
 *     the 16 luma sub-blocks, contextually probability-coded against
 *     the modes of the immediate above + left sub-blocks
 *   - the chroma prediction mode (DC/V/H/TM)
 *
 * For non-keyframes there's an additional segment-id and skip-coef
 * field, but WebP only carries keyframes so we don't implement those.
 *
 * Reference: RFC 6386 §11, §16.
 */
import type { BoolDecoder } from './bool-decoder'
import {
  B_DC_PRED,
  B_HE_PRED,
  B_HU_PRED,
  B_HD_PRED,
  B_LD_PRED,
  B_PRED,
  B_RD_PRED,
  B_TM_PRED,
  B_VE_PRED,
  B_VL_PRED,
  B_VR_PRED,
  B_MODE_TREE,
  DC_PRED,
  H_PRED,
  KF_UV_MODE_PROBS,
  KF_Y_MODE_PROBS,
  KF_Y_MODE_TREE,
  TM_PRED,
  UV_MODE_TREE,
  V_PRED,
  kfBmodeProb,
} from './tables'

/**
 * Walk a probability-coded tree. `tree` is the flat tree array (pairs
 * of (left,right) where negative values are leaf modes encoded as
 * `-(mode+1)`-but-actually... see below). `probs` is the probability
 * vector keyed by node index. Returns the leaf mode.
 *
 * VP8's tree format: at each internal node, read one bit with
 * `probs[node>>1]`, follow `tree[node + bit]`. If the followed value
 * is non-positive it's a leaf returning `-leaf` (so the leaf code is
 * stored as a negative integer to disambiguate from indices).
 */
function readTree(bool: BoolDecoder, tree: Int8Array, probs: Uint8Array | ArrayLike<number>): number {
  let i = 0
  while (true) {
    const bit = bool.readBit(probs[i >> 1])
    const next = tree[i + bit]
    if (next <= 0) return -next
    i = next
  }
}

/**
 * Decode the macroblock-level Y prediction mode for a keyframe.
 * Returns one of `DC_PRED`, `V_PRED`, `H_PRED`, `TM_PRED`, `B_PRED`.
 */
export function decodeKeyframeYMode(bool: BoolDecoder): number {
  return readTree(bool, KF_Y_MODE_TREE, KF_Y_MODE_PROBS)
}

/**
 * Decode the chroma prediction mode for a keyframe macroblock.
 * Returns one of `DC_PRED`, `V_PRED`, `H_PRED`, `TM_PRED`.
 */
export function decodeKeyframeUVMode(bool: BoolDecoder): number {
  return readTree(bool, UV_MODE_TREE, KF_UV_MODE_PROBS)
}

/**
 * Decode the 16 per-4×4 sub-block prediction modes for a B_PRED
 * macroblock, given the modes of the row of sub-blocks immediately
 * above (`aboveModes`, length 4) and the column to the left
 * (`leftModes`, length 4). Output is filled into `out` (length 16),
 * indexed in raster order (block 0 = top-left, block 15 = bottom-right).
 *
 * `aboveModes[i]` and `leftModes[i]` are mutated as we go so the next
 * macroblock can pick up the right boundary modes (we leave the
 * caller to copy out the relevant bottom row / right column).
 */
export function decodeBPredModes(
  bool: BoolDecoder,
  aboveModes: Uint8Array,
  leftModes: Uint8Array,
  out: Uint8Array,
): void {
  for (let y = 0; y < 4; y++) {
    let leftMode = leftModes[y]
    for (let x = 0; x < 4; x++) {
      const above = aboveModes[x]
      const left = leftMode
      const mode = readBmode(bool, above, left)
      out[y * 4 + x] = mode
      aboveModes[x] = mode
      leftMode = mode
    }
    leftModes[y] = leftMode
  }
}

/**
 * Decode a single 4×4 intra mode, contextually probability-coded
 * against the (above, left) neighbour-modes via `kfBmodeProb`.
 */
function readBmode(bool: BoolDecoder, above: number, left: number): number {
  // The B-mode tree has 10 leaves and uses 9 probabilities (one per
  // internal node). `kfBmodeProb(above, left, idx)` produces the
  // probability vector for this neighbour pair.
  const probs = [
    kfBmodeProb(above, left, 0),
    kfBmodeProb(above, left, 1),
    kfBmodeProb(above, left, 2),
    kfBmodeProb(above, left, 3),
    kfBmodeProb(above, left, 4),
    kfBmodeProb(above, left, 5),
    kfBmodeProb(above, left, 6),
    kfBmodeProb(above, left, 7),
    kfBmodeProb(above, left, 8),
  ]
  return readTree(bool, /* B_MODE_TREE */ B_MODE_TREE, probs)
}

/** "B-mode" sentinel for a sub-block taken from a non-B_PRED macroblock.
 * When a non-B_PRED MB sits above or to the left of a B_PRED MB, its
 * 4×4 sub-blocks are treated as having a single "implied" B-mode based
 * on the macroblock's 16×16 mode. This map is from §11.6. */
export function impliedBmode(mbMode: number): number {
  switch (mbMode) {
    case DC_PRED: return B_DC_PRED
    case V_PRED: return B_VE_PRED
    case H_PRED: return B_HE_PRED
    case TM_PRED: return B_TM_PRED
    default: return B_DC_PRED // includes B_PRED itself, which shouldn't reach here
  }
}

// Re-export mode constants so consumers don't need to pull them from tables.
export {
  B_DC_PRED, B_TM_PRED, B_VE_PRED, B_HE_PRED, B_LD_PRED,
  B_RD_PRED, B_VR_PRED, B_VL_PRED, B_HD_PRED, B_HU_PRED,
  B_PRED, DC_PRED, V_PRED, H_PRED, TM_PRED,
}
