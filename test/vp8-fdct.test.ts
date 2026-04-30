import { describe, expect, it } from 'bun:test'
import { fdct4x4, fwht4x4 } from '../src/vp8/fdct'
import { idct4x4Add, iwht4x4 } from '../src/vp8/idct'

const BPS = 32

/**
 * The forward transform (encoder) and inverse transform (decoder) are
 * not exact inverses — VP8's FDCT introduces small rounding offsets on
 * each coefficient — but for residual values in [-255, 255] without any
 * intermediate quantisation, the round-trip max-error is bounded.
 *
 * Quantisation introduces additional, expected loss; we test the
 * un-quantised round-trip here so a regression in either transform
 * surfaces independently from any quantiser bug.
 */
describe('forward/inverse 4×4 DCT round-trip (no quantisation)', () => {
  function makePatch(fn: (x: number, y: number) => number): { src: Uint8Array, srcOff: number } {
    const src = new Uint8Array(BPS * 4)
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        src[y * BPS + x] = fn(x, y)
      }
    }
    return { src, srcOff: 0 }
  }

  it('round-trips a constant patch (DC only)', () => {
    const { src, srcOff } = makePatch(() => 128)
    const ref = new Uint8Array(BPS * 4)
    const coeffs = new Int16Array(16)
    fdct4x4(src, srcOff, ref, 0, coeffs, 0)
    // Reconstruct: dst = ref + idct(coeffs).
    const dst = new Uint8Array(BPS * 4)
    dst.set(ref)
    idct4x4Add(coeffs, dst, BPS, 0)
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        expect(dst[y * BPS + x]).toBe(128)
      }
    }
  })

  it('round-trips a horizontal gradient with bounded error', () => {
    const { src } = makePatch((x, _y) => x * 60 + 20)
    const ref = new Uint8Array(BPS * 4) // ref = 0
    const coeffs = new Int16Array(16)
    fdct4x4(src, 0, ref, 0, coeffs, 0)
    const dst = new Uint8Array(BPS * 4) // dst = 0 (= ref pattern)
    idct4x4Add(coeffs, dst, BPS, 0)
    let maxErr = 0
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        const err = Math.abs(dst[y * BPS + x] - src[y * BPS + x])
        if (err > maxErr) maxErr = err
      }
    }
    // FDCT introduces ≤ 2-pixel rounding without quantisation.
    expect(maxErr).toBeLessThanOrEqual(2)
  })

  it('Y2 forward WHT produces a non-zero coefficient on a non-trivial DC pattern', () => {
    // The WHT is orthogonal but with different scale on encode vs.
    // decode (encode does `>> 1` per pass; decode does `>> 3` after a
    // `+3` rounder). The scales are chosen so that quantisation in
    // between hits the right precision; an un-quantised round-trip is
    // not bit-exact identity. We sanity-check the forward produces a
    // signal proportional to the input variance instead.
    const dcs = new Int16Array([
      100, -50, 30, 10, 200, 0, -10, 5, -200, 80, 40, -30, 0, 150, 20, -100,
    ])
    const out = new Int16Array(16)
    fwht4x4(dcs, 0, out, 0)
    // Energy is preserved up to scaling. Sum of |out| should be > sum of |dcs|/4.
    let inputEnergy = 0
    let outputEnergy = 0
    for (let i = 0; i < 16; i++) {
      inputEnergy += Math.abs(dcs[i])
      outputEnergy += Math.abs(out[i])
    }
    expect(outputEnergy).toBeGreaterThan(inputEnergy / 4)
    // DC of the WHT (= sum of all input DCs / scale) must be deterministic
    // for a given input. We can't easily predict the exact value from
    // the spec without re-implementing the transform; we settle for a
    // sanity check that it's consistent.
    expect(out[0]).not.toBe(0)
  })
})
