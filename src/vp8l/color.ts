/**
 * VP8L color transform (transform type 1).
 *
 * Decorrelates the red and blue channels using green and red as
 * predictors, on a per-block basis. The transform image stores three
 * 8-bit signed coefficients per block:
 *
 *   greenToRed   — how much of `green` to subtract from `red`
 *   greenToBlue  — how much of `green` to subtract from `blue`
 *   redToBlue    — how much of `red`   to subtract from `blue`
 *
 * Forward (encoder):
 *   newRed  = red  − colorDelta(green, greenToRed)
 *   newBlue = blue − colorDelta(green, greenToBlue)
 *                  − colorDelta(newRed, redToBlue)
 *
 * Inverse (decoder), reversing the same dependency order:
 *   red  = newRed  + colorDelta(green, greenToRed)
 *   blue = newBlue + colorDelta(green, greenToBlue)
 *                  + colorDelta(red, redToBlue)
 *
 * Where `colorDelta(c, t) = ((int8_t)c × (int8_t)t) >> 5` (per-channel,
 * with c reinterpreted as a signed 8-bit value).
 *
 * The transform stores its coefficient image as a sub-image whose
 * channels carry the three coefficients:
 *   red channel   → greenToRed
 *   green channel → greenToBlue
 *   blue channel  → redToBlue
 *
 * Reference: WebP Lossless Bitstream Specification §3.2.
 */

/** `(c * t) >> 5` interpreting both as int8 then truncating to int. */
function colorDelta(c: number, t: number): number {
  // Sign-extend `c` and `t` from 8-bit two's-complement.
  const cs = (c << 24) >> 24
  const ts = (t << 24) >> 24
  return (cs * ts) >> 5
}

/** Sign-extend an 8-bit value (treat 0..127 as itself, 128..255 as negative). */
// (Inline `(v << 24) >> 24` is what the spec calls for; no helper needed.)

/**
 * Apply the inverse color transform: read the residual ARGB image and
 * the per-block coefficients, produce decoded ARGB. Mutates `argb` in
 * place because the caller hands us the post-pixel-decode buffer.
 */
export function inverseColorTransform(
  argb: Uint32Array,
  width: number,
  height: number,
  bits: number,
  ctImage: Uint32Array,
  ctWidth: number,
): void {
  for (let y = 0; y < height; y++) {
    const blockY = y >>> bits
    for (let x = 0; x < width; x++) {
      const blockX = x >>> bits
      const ctPixel = ctImage[blockY * ctWidth + blockX]
      // ctPixel layout: alpha=0xFF, red=greenToRed, green=greenToBlue, blue=redToBlue
      const greenToRed = (ctPixel >>> 16) & 0xFF
      const greenToBlue = (ctPixel >>> 8) & 0xFF
      const redToBlue = ctPixel & 0xFF

      const i = y * width + x
      const px = argb[i]
      const a = px & 0xFF000000
      const newRed = (px >>> 16) & 0xFF
      const green = (px >>> 8) & 0xFF
      const newBlue = px & 0xFF

      const red = (newRed + colorDelta(green, greenToRed)) & 0xFF
      const blue = (newBlue + colorDelta(green, greenToBlue) + colorDelta(red, redToBlue)) & 0xFF

      argb[i] = (a | (red << 16) | (green << 8) | blue) >>> 0
    }
  }
}

/**
 * Forward color transform. Picks per-block coefficients by trying a
 * small grid of `(greenToRed, greenToBlue, redToBlue)` candidates and
 * keeping whichever minimises the sum of absolute residual values
 * across the block. Mutates `argb` in place to hold residuals; returns
 * the coefficient image to be encoded as a sub-image.
 *
 * The candidate grid is chosen to cover the practically-useful range
 * `[-16, 16]` in steps of 8 — narrower than libwebp's exhaustive search
 * but enough to capture the typical greenToRed ≈ 1, greenToBlue ≈ 1.5
 * relationship in natural images. Doubling the grid would buy a few
 * more percent compression at 8× the encode time; we trade it off
 * for speed.
 */
export function applyColorTransform(
  argb: Uint32Array,
  width: number,
  height: number,
  bits: number,
): { ctImage: Uint32Array, ctWidth: number, ctHeight: number } {
  const blockSize = 1 << bits
  const ctWidth = (width + blockSize - 1) >>> bits
  const ctHeight = (height + blockSize - 1) >>> bits
  const ctImage = new Uint32Array(ctWidth * ctHeight)

  // Candidate values for each coefficient, as int8 (we'll cast to byte).
  // Real encoders search over [-128, 127]; we use a coarse grid for speed.
  const CANDIDATES = [-16, -8, 0, 8, 16]

  // Working buffer for the original block contents — we read original
  // pixels for cost estimation, then mutate argb to residuals after
  // picking the best coefficient.
  const block = new Uint32Array(blockSize * blockSize)

  for (let by = 0; by < ctHeight; by++) {
    for (let bx = 0; bx < ctWidth; bx++) {
      const x0 = bx << bits
      const y0 = by << bits
      const xEnd = Math.min(x0 + blockSize, width)
      const yEnd = Math.min(y0 + blockSize, height)

      // Snapshot the block. We need original values to score candidates
      // and to compute the final residual.
      let blockLen = 0
      for (let y = y0; y < yEnd; y++) {
        for (let x = x0; x < xEnd; x++) {
          block[blockLen++] = argb[y * width + x]
        }
      }

      let bestG2R = 0
      let bestG2B = 0
      let bestR2B = 0
      let bestCost = Infinity

      // Stage 1: pick `greenToRed` minimising `|red - colorDelta(green, g2r)|`.
      // The three coefficients are nearly independent in their effect on
      // the residual, so we optimise them sequentially. This is what
      // libwebp does too.
      for (const g2r of CANDIDATES) {
        let cost = 0
        for (let i = 0; i < blockLen; i++) {
          const px = block[i]
          const r = (px >>> 16) & 0xFF
          const g = (px >>> 8) & 0xFF
          const newRed = (r - colorDelta(g, g2r)) & 0xFF
          // `(newRed << 24) >> 24` re-interprets a uint8 residual as
          // int8 so cost penalises both 1 and 255 equivalently.
          cost += Math.abs((newRed << 24) >> 24)
        }
        if (cost < bestCost) { bestCost = cost; bestG2R = g2r }
      }

      // Stage 2: pick `(greenToBlue, redToBlue)` jointly. Per spec, the
      // `redToBlue` term scales the *original* red — not the residual.
      // (Confusingly, the inverse on the decoder side uses the original
      // red too, since that's what the inverse-red step recovered.)
      bestCost = Infinity
      for (const g2b of CANDIDATES) {
        for (const r2b of CANDIDATES) {
          let cost = 0
          for (let i = 0; i < blockLen; i++) {
            const px = block[i]
            const r = (px >>> 16) & 0xFF
            const g = (px >>> 8) & 0xFF
            const b = px & 0xFF
            const newBlue = (b - colorDelta(g, g2b) - colorDelta(r, r2b)) & 0xFF
            cost += Math.abs((newBlue << 24) >> 24)
          }
          if (cost < bestCost) { bestCost = cost; bestG2B = g2b; bestR2B = r2b }
        }
      }

      // Encode coefficients into the CT-image pixel.
      ctImage[by * ctWidth + bx] = (
        (0xFF << 24)
        | ((bestG2R & 0xFF) << 16)
        | ((bestG2B & 0xFF) << 8)
        | (bestR2B & 0xFF)
      ) >>> 0

      // Apply the chosen transform to the actual argb buffer.
      for (let y = y0; y < yEnd; y++) {
        for (let x = x0; x < xEnd; x++) {
          const i = y * width + x
          const px = argb[i]
          const a = px & 0xFF000000
          const r = (px >>> 16) & 0xFF
          const g = (px >>> 8) & 0xFF
          const b = px & 0xFF
          const newRed = (r - colorDelta(g, bestG2R)) & 0xFF
          const newBlue = (b - colorDelta(g, bestG2B) - colorDelta(r, bestR2B)) & 0xFF
          argb[i] = (a | (newRed << 16) | (g << 8) | newBlue) >>> 0
        }
      }
    }
  }

  return { ctImage, ctWidth, ctHeight }
}
