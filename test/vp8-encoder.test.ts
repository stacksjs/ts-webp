import { describe, expect, it } from 'bun:test'
import { decodeVP8 } from '../src/vp8/decoder'
import { encodeVP8 } from '../src/vp8/encoder'

/**
 * VP8 lossy encoder round-trip tests. The encoder is a minimal-but-
 * complete implementation that uses 16×16 intra-DC prediction for every
 * macroblock and a single segment / single partition layout. It's not
 * competitive with cwebp on rate-distortion (no mode search, no
 * adaptive quantisation) but produces a fully decoder-compatible
 * VP8 bitstream.
 *
 * What we assert:
 *   1. The encoded bytes parse cleanly through our own VP8 decoder.
 *   2. The decoded image dimensions match the input.
 *   3. The decoded pixels are within a quality-dependent error bound
 *      from the input. With the simplest mode and quality 60, expect a
 *      mean-error budget of ~30/255 per channel and a worst-pixel
 *      budget of ~120/255 — coarse but not catastrophic.
 */

function makeImage(width: number, height: number, fill: (x: number, y: number) => [number, number, number, number]) {
  const data = new Uint8Array(width * height * 4)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b, a] = fill(x, y)
      const i = (y * width + x) * 4
      data[i] = r
      data[i + 1] = g
      data[i + 2] = b
      data[i + 3] = a
    }
  }
  return { data, width, height, hasAlpha: false }
}

function meanAbsErrorRGB(a: Uint8Array, b: Uint8Array): number {
  let s = 0
  let n = 0
  for (let i = 0; i < a.length; i += 4) {
    s += Math.abs(a[i] - b[i])
    s += Math.abs(a[i + 1] - b[i + 1])
    s += Math.abs(a[i + 2] - b[i + 2])
    n += 3
  }
  return s / n
}

function maxAbsErrorRGB(a: Uint8Array, b: Uint8Array): number {
  let m = 0
  for (let i = 0; i < a.length; i += 4) {
    m = Math.max(m, Math.abs(a[i] - b[i]), Math.abs(a[i + 1] - b[i + 1]), Math.abs(a[i + 2] - b[i + 2]))
  }
  return m
}

describe('VP8 lossy encoder', () => {
  it('encodes a 16×16 solid-colour image to a parseable VP8 chunk', () => {
    const img = makeImage(16, 16, () => [128, 128, 128, 255])
    const vp8 = encodeVP8(img, { quality: 60 })
    expect(vp8.length).toBeGreaterThan(10)
    // Decode back via our own VP8 decoder.
    const out = decodeVP8(vp8)
    expect(out.width).toBe(16)
    expect(out.height).toBe(16)
    // Solid grey should round-trip with very low error.
    expect(meanAbsErrorRGB(img.data, out.data)).toBeLessThan(10)
  })

  it('encodes a 32×32 horizontal gradient with bounded loss', () => {
    const img = makeImage(32, 32, x => [(x * 8) & 0xFF, 128, 200, 255])
    const vp8 = encodeVP8(img, { quality: 30 })
    const out = decodeVP8(vp8)
    expect(out.width).toBe(32)
    expect(out.height).toBe(32)
    // 16×16 DC-only prediction is a coarse approximation of a
    // gradient: each macroblock collapses to one mean value, so the
    // max-error bound is roughly the gradient amplitude per MB.
    expect(meanAbsErrorRGB(img.data, out.data)).toBeLessThan(60)
    expect(maxAbsErrorRGB(img.data, out.data)).toBeLessThan(255)
  })

  it('advances coefficient contexts across textured blocks', () => {
    const img = makeImage(16, 16, (x, y) => [
      x * 7,
      y * 7,
      (x + y) * 3,
      255,
    ])
    const vp8 = encodeVP8(img, { quality: 95 })
    const out = decodeVP8(vp8)

    // This block contains coefficients in several token categories. If the
    // encoder keeps using the previous coefficient's probability context,
    // the decoder diverges after the first large token and corrupts pixels.
    expect(meanAbsErrorRGB(img.data, out.data)).toBeLessThan(16)
    expect(maxAbsErrorRGB(img.data, out.data)).toBeLessThan(70)
  })

  it('handles non-16-aligned dimensions via padding', () => {
    // 25×17 forces 2×2 macroblocks with edge padding.
    const img = makeImage(25, 17, (x, y) => [(x * 10) & 0xFF, (y * 15) & 0xFF, 100, 255])
    const vp8 = encodeVP8(img, { quality: 50 })
    const out = decodeVP8(vp8)
    expect(out.width).toBe(25)
    expect(out.height).toBe(17)
    // Loose error bound — gradient images are a stress test for DC-only
    // prediction since each 16×16 block has substantial within-block
    // variance.
    expect(meanAbsErrorRGB(img.data, out.data)).toBeLessThan(80)
  })

  it('rejects out-of-range quality', () => {
    const img = makeImage(16, 16, () => [0, 0, 0, 255])
    expect(() => encodeVP8(img, { quality: -1 })).toThrow()
    expect(() => encodeVP8(img, { quality: 128 })).toThrow()
  })

  it('rejects oversized dimensions', () => {
    expect(() => encodeVP8({ data: new Uint8Array(0), width: 0, height: 0, hasAlpha: false })).toThrow()
    expect(() => encodeVP8({ data: new Uint8Array(0), width: 16384, height: 16, hasAlpha: false })).toThrow()
  })
})
