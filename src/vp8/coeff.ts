/**
 * VP8 DCT coefficient decoder.
 *
 * Each macroblock holds 24 or 25 4×4 sub-blocks of quantised DCT
 * coefficients (16 luma + 4 U + 4 V; the 25th block is the Y2 WHT block
 * holding the 16 luma DCs when the macroblock uses 16×16 prediction).
 * Coefficients are arithmetic-coded with context-dependent probability
 * tables; this module walks `DCT_TOKEN_TREE` per coefficient using
 * `BoolDecoder` and produces a 16-entry signed integer array per block
 * (still in zig-zag order — IDCT will re-order them).
 *
 * Reference: RFC 6386 §13.
 */
import { BoolDecoder } from './bool-decoder'
import {
  COEF_BANDS,
  DCT_CAT_BASE,
  DCT_CAT_PROBS,
  DEFAULT_COEF_PROBS,
  NUM_COEF_BANDS,
} from './tables'

/** Block types (RFC 6386 §13.3). */
export const BLOCK_TYPE_Y_AFTER_Y2 = 0 // 16 luma blocks when 16×16 mode (DCs stripped)
export const BLOCK_TYPE_Y2 = 1 // the WHT block of luma DCs
export const BLOCK_TYPE_UV = 2 // 8 chroma blocks
export const BLOCK_TYPE_Y_NO_Y2 = 3 // 16 luma blocks when B_PRED mode (full DC+AC)

/**
 * Decode the DCT coefficients of one 4×4 block. `ctxL`/`ctxA` are the
 * "non-zero" context flags from the immediate left and above blocks
 * (1 if that block had any non-zero coefficients, 0 otherwise). `out`
 * is filled with the 16 quantised coefficients in zig-zag scan order;
 * positions past the EOB are zeroed. Returns the new non-zero context
 * (1 if any coef was non-zero, 0 otherwise) for the next block to use.
 *
 * `firstCoef` is 0 for blocks that include the DC term, 1 for luma
 * blocks of a 16×16-predicted macroblock (whose DC lives in the Y2
 * block instead).
 */
export function decodeBlockCoeffs(
  bool: BoolDecoder,
  out: Int16Array,
  blockType: number,
  ctxL: number,
  ctxA: number,
  firstCoef: number,
  probs: Uint8Array = DEFAULT_COEF_PROBS,
): number {
  // Clear destination — `out` may be reused across blocks.
  for (let i = 0; i < 16; i++) out[i] = 0

  let ctx = ctxL + ctxA
  let i = firstCoef
  let prevWasZero = false
  let nonZero = 0

  while (i < 16) {
    const band = COEF_BANDS[i]
    // Probability set for this (block_type, band, ctx). 11 entries.
    const pBase = blockType * NUM_COEF_BANDS * 3 * 11 + band * 3 * 11 + ctx * 11

    // After a zero, the EOB probability is skipped: the encoder knows
    // the EOB was already ruled out (else why keep coding?). Per the
    // spec we still read prob[0] for the *first* coefficient of a
    // block but skip it if the previous coef was zero.
    if (!prevWasZero) {
      // Bit 0: end-of-block?
      if (bool.readBit(probs[pBase + 0]) === 0) break
    }

    // Bit 1: zero?
    if (bool.readBit(probs[pBase + 1]) === 0) {
      prevWasZero = true
      ctx = 0 // "zero" context for the next coef
      i++
      continue
    }

    // Non-zero from here. Decode the magnitude.
    let value: number
    if (bool.readBit(probs[pBase + 2]) === 0) {
      value = 1
    }
    else {
      // > 1 — disambiguate among 2..4 / CAT1 / CAT2 / CAT3-6.
      if (bool.readBit(probs[pBase + 3]) === 0) {
        if (bool.readBit(probs[pBase + 4]) === 0) {
          value = 2
        }
        else {
          value = bool.readBit(probs[pBase + 5]) === 0 ? 3 : 4
        }
      }
      else {
        // CAT1..CAT6.
        if (bool.readBit(probs[pBase + 6]) === 0) {
          // CAT1 or CAT2.
          if (bool.readBit(probs[pBase + 7]) === 0) {
            value = readCat(bool, 0)
          }
          else {
            value = readCat(bool, 1)
          }
        }
        else {
          // CAT3..CAT6.
          if (bool.readBit(probs[pBase + 8]) === 0) {
            // CAT3 or CAT4.
            if (bool.readBit(probs[pBase + 9]) === 0) {
              value = readCat(bool, 2)
            }
            else {
              value = readCat(bool, 3)
            }
          }
          else {
            // CAT5 or CAT6.
            if (bool.readBit(probs[pBase + 10]) === 0) {
              value = readCat(bool, 4)
            }
            else {
              value = readCat(bool, 5)
            }
          }
        }
      }
    }

    // Sign bit (uniform prob 128).
    if (bool.readBit(128) === 1) value = -value

    out[i] = value
    nonZero = 1
    prevWasZero = false
    // For the next coef, the "previous-coef magnitude" context is 1
    // if |value|==1, else 2.
    ctx = Math.abs(value) === 1 ? 1 : 2
    i++
  }

  return nonZero
}

/**
 * Decode the magnitude of a DCT_CATn token (n = 0..5, i.e. CAT1..CAT6).
 * Uses the per-bit probability tables in `DCT_CAT_PROBS`. The result
 * is unsigned; the caller applies the sign bit.
 */
function readCat(bool: BoolDecoder, catIdx: number): number {
  const probs = DCT_CAT_PROBS[catIdx]
  let v = 0
  for (let j = 0; j < probs.length; j++) {
    v = (v << 1) | bool.readBit(probs[j])
  }
  return DCT_CAT_BASE[catIdx + 6] + v
}
