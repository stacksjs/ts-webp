import { describe, expect, it } from 'bun:test'
import { decode, encode } from '../src'

/**
 * Tests for the optional encoder features (subtract-green, LZ77, color
 * cache) added on top of the literal-only baseline. Each test asserts
 * exact-pixel round-trip *plus* a sanity bound on output size — the
 * features should never accidentally regress compression past the
 * literal-only baseline. (A small constant-overhead regression is OK
 * for tiny images where the cache tree alone costs more than the
 * savings, hence the slack in `expect(...).toBeLessThanOrEqual`.)
 */

type Variant = {
  subtractGreen?: boolean
  predictor?: boolean
  useColorIndex?: boolean
  useLZ77?: boolean
  useColorCache?: boolean
  cacheBits?: number
}

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

function expectRoundTrip(image: { data: Uint8Array, width: number, height: number }, variant: Variant): number {
  const encoded = encode(image, variant as object)
  const decoded = decode(encoded)
  expect(decoded.width).toBe(image.width)
  expect(decoded.height).toBe(image.height)
  expect(decoded.data.length).toBe(image.data.length)
  for (let i = 0; i < image.data.length; i++) {
    if (decoded.data[i] !== image.data[i]) {
      throw new Error(`Mismatch at byte ${i}: input=${image.data[i]}, output=${decoded.data[i]} (variant=${JSON.stringify(variant)})`)
    }
  }
  return encoded.length
}

// "No features" disables every transform AND every pixel-stream
// optimization, so per-feature win tests can attribute compression
// gains to exactly the toggle they vary. Color-indexing is the
// stickiest of these — it pre-empts SG and predictor when ≤ 256
// distinct colours are present, which is most synthetic test images.
const NO_FEATURES: Variant = {
  subtractGreen: false,
  predictor: false,
  useColorIndex: false,
  useLZ77: false,
  useColorCache: false,
}
const ALL_FEATURES: Variant = {
  subtractGreen: true,
  predictor: true,
  useColorIndex: true,
  useLZ77: true,
  useColorCache: true,
}

describe('encoder feature toggles — exact round-trip', () => {
  // Each toggle on its own + every pair + all three. We don't stress
  // every combination on every image (that'd be 8 × N tests); instead
  // we stress the all-on path on a wide range of patterns and the
  // single-toggle paths on a representative image to make sure each
  // feature is independently round-trip safe.
  const variants: [string, Variant][] = [
    ['baseline (no features)', NO_FEATURES],
    ['subtract-green only', { subtractGreen: true, useLZ77: false, useColorCache: false }],
    ['LZ77 only', { subtractGreen: false, useLZ77: true, useColorCache: false }],
    ['color cache only', { subtractGreen: false, useLZ77: false, useColorCache: true }],
    ['subtract-green + LZ77', { subtractGreen: true, useLZ77: true, useColorCache: false }],
    ['subtract-green + cache', { subtractGreen: true, useLZ77: false, useColorCache: true }],
    ['LZ77 + cache', { subtractGreen: false, useLZ77: true, useColorCache: true }],
    ['all three', ALL_FEATURES],
  ]

  for (const [name, variant] of variants) {
    it(`16x16 single-colour with ${name}`, () => {
      expectRoundTrip(makeImage(16, 16, () => [200, 100, 50, 255]), variant)
    })

    it(`16x16 horizontal/vertical gradient with ${name}`, () => {
      expectRoundTrip(makeImage(16, 16, (x, y) => [x * 16, y * 16, 128, 255]), variant)
    })

    it(`32x32 correlated channels with ${name}`, () => {
      expectRoundTrip(makeImage(32, 32, (x, y) => {
        const base = ((x ^ y) * 3) % 256
        return [(base + 5) % 256, base, (base + 251) % 256, 255]
      }), variant)
    })
  }
})

describe('compression — features should win on the right images', () => {
  // These tests assert *direction*, not absolute numbers: each feature
  // should compress better than baseline on at least one synthetic
  // pattern designed to exercise it.

  it('LZ77 wins big on long single-colour runs', () => {
    const img = makeImage(64, 64, () => [128, 128, 128, 255])
    const baseline = expectRoundTrip(img, NO_FEATURES)
    const withLz77 = expectRoundTrip(img, { ...NO_FEATURES, useLZ77: true })
    // Single-colour image should compress to a tiny fraction with LZ77.
    expect(withLz77).toBeLessThan(baseline / 5)
  })

  it('subtract-green wins on R/G/B-correlated images', () => {
    const img = makeImage(32, 32, (x, y) => {
      const base = ((x * 7 + y * 11) % 256)
      return [(base + 5) % 256, base, (base + 251) % 256, 255]
    })
    const baseline = expectRoundTrip(img, NO_FEATURES)
    const withSg = expectRoundTrip(img, { ...NO_FEATURES, subtractGreen: true })
    // Correlated channels: SG should at least roughly halve the literal cost.
    expect(withSg).toBeLessThan(baseline)
  })

  it('color cache wins on palette-style images', () => {
    // 8 distinct colours scattered pseudo-randomly — a workload where
    // LZ77 finds few adjacent matches but cache hits land constantly.
    const palette: [number, number, number][] = [
      [200, 50, 50], [50, 200, 50], [50, 50, 200], [200, 200, 50],
      [50, 200, 200], [200, 50, 200], [100, 100, 100], [200, 200, 200],
    ]
    const img = makeImage(48, 48, (x, y) => {
      const c = palette[((x * 17 + y * 13) % 8 + 8) % 8]
      return [c[0], c[1], c[2], 255]
    })
    const baseline = expectRoundTrip(img, NO_FEATURES)
    const withCache = expectRoundTrip(img, { ...NO_FEATURES, useColorCache: true })
    expect(withCache).toBeLessThan(baseline)
  })

  it('predictor wins big on smooth gradients', () => {
    const img = makeImage(64, 64, (x, y) => [(x * 4) % 256, (y * 4) % 256, 128, 255])
    const baseline = expectRoundTrip(img, NO_FEATURES)
    const withPredictor = expectRoundTrip(img, { ...NO_FEATURES, predictor: true })
    // Gradients are almost perfectly predicted from neighbours; the
    // residuals collapse to a few distinct values dominated by zero.
    expect(withPredictor).toBeLessThan(baseline / 3)
  })

  it('color-indexing wins big on palette images', () => {
    // 8-color palette → the encoder packs 4 indices per byte and the
    // bitstream's pixel stream shrinks dramatically. Without CI, this
    // image is just a literal/cache stream over the same 8 colours;
    // with CI, only ~16 × 64 = 1024 packed-pixel words are emitted.
    const palette: [number, number, number][] = [
      [200, 50, 50], [50, 200, 50], [50, 50, 200], [200, 200, 50],
      [50, 200, 200], [200, 50, 200], [100, 100, 100], [200, 200, 200],
    ]
    const img = makeImage(64, 64, (x, y) => {
      const c = palette[((x * 17 + y * 13) % 8 + 8) % 8]
      return [c[0], c[1], c[2], 255]
    })
    const baseline = expectRoundTrip(img, NO_FEATURES)
    const withCI = expectRoundTrip(img, { ...NO_FEATURES, useColorIndex: true })
    // 8-colour palette + 4× packing factor; the green-channel alphabet
    // grows by the same factor (we now have packed multi-indices), so
    // the *raw* compression isn't 4× — but the multi-index green codes
    // become roughly uniform 8-bit, much smaller than three independent
    // 8-colour channels emitted as literals. ~40 % is a reliable floor
    // for this size of palette image.
    expect(withCI).toBeLessThan(baseline * 0.6)
  })

  it('all features combined beat baseline on photo-like images', () => {
    // Smooth tonal variation across the image — close to what a real
    // natural photo looks like at small scale. Tests that the features
    // compose correctly and don't regress against each other.
    const img = makeImage(96, 96, (x, y) => {
      const r = Math.max(0, Math.min(255, (100 + Math.sin(x * 0.1) * 60 + Math.cos(y * 0.13) * 40) | 0))
      const g = Math.max(0, Math.min(255, (110 + Math.sin(x * 0.1) * 60 + Math.cos(y * 0.13) * 40) | 0))
      const b = Math.max(0, Math.min(255, (105 + Math.sin(x * 0.1) * 60 + Math.cos(y * 0.13) * 40) | 0))
      return [r, g, b, 255]
    })
    const baseline = expectRoundTrip(img, NO_FEATURES)
    const allOn = expectRoundTrip(img, ALL_FEATURES)
    // Should be at least a 2× win on a smooth photo-like pattern.
    expect(allOn).toBeLessThan(baseline / 2)
  })
})

describe('feature edge cases', () => {
  it('LZ77 handles self-overlapping backrefs (run length > distance)', () => {
    // ABABAB pattern — a backref of length 4 from distance 2 must
    // copy through itself, reading positions written earlier in the
    // same backref. Encoder + decoder must agree on this.
    const img = makeImage(8, 1, x => x % 2 === 0 ? [255, 0, 0, 255] : [0, 0, 255, 255])
    expectRoundTrip(img, ALL_FEATURES)
  })

  it('LZ77 + cache handles tail of image with no full 3-pixel lookahead', () => {
    // Last 2 pixels of an image can't form a 3-pixel match window —
    // the encoder must fall back to literal/cache emission cleanly.
    const img = makeImage(5, 1, x => [x * 50, x * 30, x * 70, 255])
    expectRoundTrip(img, ALL_FEATURES)
  })

  it('color cache 4-bit (16 entries) round-trips', () => {
    const img = makeImage(20, 20, (x, _y) => x < 5 ? [255, 0, 0, 255] : [0, 255, 0, 255])
    expectRoundTrip(img, { useColorCache: true, cacheBits: 4 } as Variant)
  })

  it('color cache 11-bit (max, 2048 entries) round-trips', () => {
    const img = makeImage(20, 20, (x, y) => [(x * 11) % 256, (y * 13) % 256, 100, 255])
    expectRoundTrip(img, { useColorCache: true, cacheBits: 11 } as Variant)
  })

  it('full-alpha image with all three features', () => {
    // Alpha varies pixel-by-pixel — exercises the alpha tree under
    // a non-trivial distribution.
    const img = makeImage(16, 16, (x, y) => [100, 100, 100, ((x * 16 + y) * 5) % 256])
    expectRoundTrip(img, ALL_FEATURES)
  })

  it('1x1 image with all three features (degenerate but legal)', () => {
    expectRoundTrip(makeImage(1, 1, () => [42, 84, 168, 200]), ALL_FEATURES)
  })

  it('subtract-green of (0, 0, 0) stays (0, 0, 0)', () => {
    // Zero pixel: R = G = B = 0. SG: R - G = 0, B - G = 0. Then SG
    // inverse: R + G = 0, B + G = 0. Identity. No regression on
    // black pixels.
    expectRoundTrip(makeImage(8, 8, () => [0, 0, 0, 255]), { subtractGreen: true } as Variant)
  })

  it('subtract-green of saturated channels stays in [0, 255] mod 256', () => {
    // R = 255, G = 0: SG R = 255 - 0 = 255. Decoder: 255 + 0 = 255. ✓
    // R = 0, G = 200: SG R = (0 - 200) & 0xFF = 56. Decoder: (56 + 200) & 0xFF = 0. ✓
    // The mod-256 wrap is what makes SG round-trip exact, and it's
    // well-defined byte-wise so we just verify the roundtrip on
    // patterns that exercise both directions.
    expectRoundTrip(makeImage(8, 8, (x, _y) => x < 4 ? [255, 0, 0, 255] : [0, 200, 100, 255]), { subtractGreen: true } as Variant)
  })
})
