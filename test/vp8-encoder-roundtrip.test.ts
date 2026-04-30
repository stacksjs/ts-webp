import { describe, expect, it } from 'bun:test'
import { decode, encode } from '../src'

/**
 * End-to-end round-trip tests through the public `encode` / `decode`
 * API for both lossless (VP8L) and lossy (VP8) paths. The lossless
 * path is bit-exact; the lossy path is bounded-error.
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
  return { data, width, height }
}

function maeRGB(a: Uint8Array, b: Uint8Array): number {
  let s = 0
  let n = 0
  for (let i = 0; i < a.length; i += 4) {
    s += Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2])
    n += 3
  }
  return s / n
}

describe('public encode / decode round-trip', () => {
  it('lossless round-trips byte-for-byte', () => {
    const img = makeImage(48, 48, (x, y) => [(x * 5) & 0xFF, (y * 7) & 0xFF, ((x + y) * 3) & 0xFF, 255])
    const enc = encode(img, { lossless: true })
    const dec = decode(enc)
    expect(dec.width).toBe(48)
    expect(dec.height).toBe(48)
    expect(dec.data).toEqual(img.data)
  })

  it('lossy round-trips with bounded RGB error', () => {
    const img = makeImage(48, 48, (x, y) => [(x * 5) & 0xFF, (y * 7) & 0xFF, ((x + y) * 3) & 0xFF, 255])
    const enc = encode(img, { lossless: false, quality: 75 })
    const dec = decode(enc)
    expect(dec.width).toBe(48)
    expect(dec.height).toBe(48)
    // 16×16 intra-DC prediction throws away a lot on a fast-changing
    // gradient like this — every macroblock collapses to one mean per
    // channel. We assert the error is bounded but loose; reach for
    // cwebp when output quality matters more than pure-TS portability.
    expect(maeRGB(img.data, dec.data)).toBeLessThan(80)
  })

  it('lossy quality knob trades size for accuracy monotonically', () => {
    const img = makeImage(48, 48, (x, y) => [
      Math.floor(128 + 64 * Math.sin(x * 0.3)),
      Math.floor(128 + 64 * Math.cos(y * 0.4)),
      Math.floor(128 + 64 * Math.sin((x + y) * 0.2)),
      255,
    ] as [number, number, number, number])
    const lo = encode(img, { lossless: false, quality: 10 })
    const hi = encode(img, { lossless: false, quality: 95 })
    // Higher quality → larger encoded size (this is how every lossy
    // codec is supposed to behave; if the relationship inverts we
    // probably have a quantiser-mapping bug).
    expect(hi.length).toBeGreaterThanOrEqual(lo.length)
    // And higher quality → lower decode error.
    const decLo = decode(lo)
    const decHi = decode(hi)
    expect(maeRGB(img.data, decHi.data)).toBeLessThanOrEqual(maeRGB(img.data, decLo.data))
  })
})
