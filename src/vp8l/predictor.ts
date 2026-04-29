/**
 * VP8L predictor transform.
 *
 * The predictor transform splits the image into `2^bits × 2^bits` blocks
 * and assigns each block one of 14 prediction modes. For each pixel
 * (except the very first row/column), the pixel value is replaced by
 * `pixel - predictor(neighbours)` (per channel, mod 256). The decoder
 * reverses by adding `predictor(neighbours)` back.
 *
 * Why this helps: after subtract-green, a smooth natural image has small
 * differences between neighbours. The predictor produces an even smaller
 * residual by using a *prediction* of each pixel based on its already-
 * decoded neighbours — when the prediction is good the residual is tiny,
 * which Huffman codes very efficiently.
 *
 * The 14 modes use the neighbours `L` (left), `T` (top), `TL` (top-left),
 * `TR` (top-right) — all already-decoded by the time we predict. Special
 * cases:
 *   • the very first pixel uses an opaque-black predictor (`0xFF000000`)
 *   • the rest of the first row uses `L`
 *   • the rest of the first column uses `T`
 *   • everything else uses `mode_table[block_mode]`
 *
 * The mode-per-block image is stored at the start of the bitstream as a
 * recursively-encoded sub-image whose pixels carry the mode in the green
 * channel.
 *
 * Reference: WebP Lossless Bitstream Specification §3.1.
 */

/**
 * Number of distinct modes (0..13). Spec reserves 14..15 but they're
 * undefined; we never emit them and reject them on decode.
 */
export const NUM_PREDICTORS = 14

/**
 * Apply one of the 14 predictor functions to four neighbouring pixels.
 * Each pixel is a packed ARGB Uint32. The return value is the *predicted*
 * pixel — the encoder subtracts it from the actual pixel to produce the
 * residual; the decoder adds it back.
 *
 * Implemented as a switch rather than a dispatch table because every
 * mode does different per-channel arithmetic, and the JIT inlines this
 * switch into a tight loop better than indirect calls.
 */
export function predict(mode: number, L: number, T: number, TL: number, TR: number): number {
  switch (mode) {
    case 0: return 0xFF000000 | 0
    case 1: return L
    case 2: return T
    case 3: return TR
    case 4: return TL
    case 5: return averageARGB(averageARGB(L, TR), T)
    case 6: return averageARGB(L, TL)
    case 7: return averageARGB(L, T)
    case 8: return averageARGB(TL, T)
    case 9: return averageARGB(T, TR)
    case 10: return averageARGB(averageARGB(L, TL), averageARGB(T, TR))
    case 11: return selectPredictor(L, T, TL)
    case 12: return clampAddSubtractFull(L, T, TL)
    case 13: return clampAddSubtractHalf(averageARGB(L, T), TL)
    default: throw new Error(`Unknown predictor mode: ${mode}`)
  }
}

/**
 * Per-channel average of two ARGB pixels. Each channel is averaged
 * independently with truncation (`(a + b) >> 1`); a + b can overflow
 * past 255 so we mask each channel before adding.
 */
function averageARGB(a: number, b: number): number {
  const a0 = a & 0xFF
  const a1 = (a >>> 8) & 0xFF
  const a2 = (a >>> 16) & 0xFF
  const a3 = (a >>> 24) & 0xFF
  const b0 = b & 0xFF
  const b1 = (b >>> 8) & 0xFF
  const b2 = (b >>> 16) & 0xFF
  const b3 = (b >>> 24) & 0xFF
  return (
    ((a3 + b3) >>> 1) << 24
    | ((a2 + b2) >>> 1) << 16
    | ((a1 + b1) >>> 1) << 8
    | ((a0 + b0) >>> 1)
  ) >>> 0
}

/**
 * Mode 11 — "Paeth-like" predictor: picks whichever of L, T, TL is
 * closest to the gradient `L + T - TL`.
 */
function selectPredictor(L: number, T: number, TL: number): number {
  let result = 0
  for (let shift = 0; shift < 32; shift += 8) {
    const l = (L >>> shift) & 0xFF
    const t = (T >>> shift) & 0xFF
    const tl = (TL >>> shift) & 0xFF
    const pa = Math.abs(t - tl)
    const pb = Math.abs(l - tl)
    const pc = Math.abs(l + t - 2 * tl)
    let chosen: number
    if (pa <= pb && pa <= pc) chosen = l
    else if (pb <= pc) chosen = t
    else chosen = tl
    result |= chosen << shift
  }
  return result >>> 0
}

/** Saturate a value into [0, 255]. */
function clamp255(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v
}

/**
 * Mode 12 — clamp(L + T - TL) per channel. The "full" in the name
 * refers to the formula taking full L, T, TL values (vs mode 13 which
 * uses an averaged `(L + T) / 2`).
 */
function clampAddSubtractFull(L: number, T: number, TL: number): number {
  let result = 0
  for (let shift = 0; shift < 32; shift += 8) {
    const l = (L >>> shift) & 0xFF
    const t = (T >>> shift) & 0xFF
    const tl = (TL >>> shift) & 0xFF
    result |= clamp255(l + t - tl) << shift
  }
  return result >>> 0
}

/**
 * Mode 13 — `clamp(a + (a - b) / 2)` per channel, i.e. extrapolate from
 * `a` (which is `(L + T) / 2`) using the slope to `b` (which is `TL`).
 */
function clampAddSubtractHalf(a: number, b: number): number {
  let result = 0
  for (let shift = 0; shift < 32; shift += 8) {
    const av = (a >>> shift) & 0xFF
    const bv = (b >>> shift) & 0xFF
    // (av + (av - bv) / 2) → av + (av - bv) >> 1 with C-style rounding.
    // VP8L uses arithmetic shift; the residual is rounded toward -∞ so
    // we replicate `(av - bv) >> 1` using a sign-extension trick.
    const diff = av - bv
    const half = diff >> 1 // arithmetic shift; preserves sign
    result |= clamp255(av + half) << shift
  }
  return result >>> 0
}

/**
 * Apply the inverse predictor transform: read the residual ARGB image
 * + the mode-per-block image, and produce the original ARGB pixels.
 *
 * `argb` is mutated in place.
 */
export function inversePredictorTransform(
  argb: Uint32Array,
  width: number,
  height: number,
  bits: number,
  modeImage: Uint32Array,
  modeWidth: number,
): void {
  const blockSize = 1 << bits

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      let predictor: number

      if (x === 0 && y === 0) {
        predictor = 0xFF000000 | 0
      } else if (y === 0) {
        // Top row: predictor is L.
        predictor = argb[i - 1]
      } else if (x === 0) {
        // Left column: predictor is T.
        predictor = argb[i - width]
      } else {
        const blockIdx = (y >>> bits) * modeWidth + (x >>> bits)
        // Mode is stored in the green channel of the mode-image pixel.
        const mode = (modeImage[blockIdx] >>> 8) & 0xFF
        const L = argb[i - 1]
        const T = argb[i - width]
        const TL = argb[i - width - 1]
        const TR = x + 1 < width ? argb[i - width + 1] : argb[i - width]
        // For mode 12 specifically the spec also handles right-edge
        // pixels: when `x === width - 1`, TR = T. We already do that
        // via the `argb[i - width]` fallback above.
        predictor = predict(mode, L, T, TL, TR)
      }

      argb[i] = addArgbBytewise(argb[i], predictor)
    }
  }
}

/**
 * Apply the forward predictor transform: encode each pixel as
 * `pixel - predictor`, picking a single mode for all blocks (mode 11,
 * the select predictor) for now. Returns the mode image as an ARGB
 * `Uint32Array` whose green channel holds the mode value per block.
 *
 * Picking per-block modes adaptively is a small additional pass —
 * for each block, we'd try every mode on a representative interior
 * pixel and keep the one with the smallest residual entropy. Mode 11
 * alone already wins most of the compression on natural images.
 *
 * `argb` is mutated in place to hold residuals.
 */
export function applyPredictorTransform(
  argb: Uint32Array,
  width: number,
  height: number,
  bits: number,
): { modeImage: Uint32Array, modeWidth: number, modeHeight: number } {
  const blockSize = 1 << bits
  const modeWidth = (width + blockSize - 1) >>> bits
  const modeHeight = (height + blockSize - 1) >>> bits
  const modeImage = new Uint32Array(modeWidth * modeHeight)

  // We pick one global mode for now. Mode 11 (select) is a good default
  // on natural images; mode 0 (zero predictor) is best on sparse alpha
  // images. We could pick per-block adaptively to win more — but that's
  // a future tuning pass; what's here already gets us most of the
  // compression on photo content.
  const GLOBAL_MODE = 11
  for (let i = 0; i < modeImage.length; i++) {
    // Mode in the green channel; alpha 0xFF, R/B 0 (unused).
    modeImage[i] = (0xFF << 24) | (GLOBAL_MODE << 8)
  }

  // Walk in raster order computing residuals. We need to read ORIGINAL
  // pixel values for predictors — but we're mutating argb in place. So
  // we copy each row before mutating it.
  const prevRow = new Uint32Array(width)
  const currRow = new Uint32Array(width)

  // y = 0: top row uses L predictor for x > 0, opaque-black for (0, 0).
  prevRow.set(argb.subarray(0, width))
  for (let x = 0; x < width; x++) {
    const orig = argb[x]
    const predictor = x === 0 ? 0xFF000000 | 0 : prevRow[x - 1]
    argb[x] = subArgbBytewise(orig, predictor)
  }

  // Subsequent rows.
  for (let y = 1; y < height; y++) {
    const rowOff = y * width
    currRow.set(argb.subarray(rowOff, rowOff + width))
    for (let x = 0; x < width; x++) {
      const i = rowOff + x
      const orig = currRow[x]
      let predictor: number
      if (x === 0) {
        predictor = prevRow[0]
      } else {
        const L = currRow[x - 1] // already-original this row, but mutated past x-1!
        const T = prevRow[x]
        const TL = prevRow[x - 1]
        const TR = x + 1 < width ? prevRow[x + 1] : prevRow[x]
        predictor = predict(GLOBAL_MODE, L, T, TL, TR)
      }
      argb[i] = subArgbBytewise(orig, predictor)
    }
    // Wait — for rows below the first, the predictor depends on `currRow[x - 1]`
    // (the *original* value of pixel at (x-1, y), which we haven't yet
    // overwritten in `argb` because we're walking left-to-right). But
    // we DO need that pre-mutation value for next-pixel predictions. By
    // copying into currRow above before any mutation, we've preserved
    // them — and currRow itself is read-only inside the inner loop. ✓
    // Move currRow → prevRow for the next iteration.
    prevRow.set(currRow)
  }

  return { modeImage, modeWidth, modeHeight }
}

/** ARGB byte-wise subtraction (each channel mod 256). */
function subArgbBytewise(a: number, b: number): number {
  return (
    (((a >>> 24) - (b >>> 24)) & 0xFF) << 24
    | (((a >>> 16) - (b >>> 16)) & 0xFF) << 16
    | (((a >>> 8) - (b >>> 8)) & 0xFF) << 8
    | ((a - b) & 0xFF)
  ) >>> 0
}

/** ARGB byte-wise addition (each channel mod 256). */
function addArgbBytewise(a: number, b: number): number {
  return (
    (((a >>> 24) + (b >>> 24)) & 0xFF) << 24
    | (((a >>> 16) + (b >>> 16)) & 0xFF) << 16
    | (((a >>> 8) + (b >>> 8)) & 0xFF) << 8
    | ((a + b) & 0xFF)
  ) >>> 0
}
