import { describe, expect, it } from 'bun:test'
import { decode, encode } from '../src/index'
import { parseRiff } from '../src/riff'

/**
 * Lossy encoding keeps alpha.
 *
 * `encode(..., { lossless: false })` used to write a bare `VP8 ` chunk and
 * drop the alpha channel without a word. Image pipelines ask for lossy by
 * default, so a transparent cutout came out as a subject on a solid black
 * box. Translucent input now becomes `VP8X + ALPH + VP8`.
 */

function cutout(size: number): { data: Uint8Array, width: number, height: number } {
  const data = new Uint8Array(size * size * 4)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4
      const inside = (x - size / 2) ** 2 + (y - size / 2) ** 2 < (size / 3) ** 2
      data[i] = 210
      data[i + 1] = 150
      data[i + 2] = 90
      // A ramp along the top rows exercises every level, not just 0 and 255.
      data[i + 3] = y < 4 ? Math.round((x / (size - 1)) * 255) : inside ? 255 : 0
    }
  }
  return { data, width: size, height: size }
}

describe('lossy WebP with alpha', () => {
  it('writes VP8X, ALPH and VP8 for a translucent image', () => {
    const encoded = encode(cutout(48), { lossless: false, quality: 80 })
    expect(parseRiff(encoded).map(chunk => chunk.fourCC)).toEqual(['VP8X', 'ALPH', 'VP8 '])
  })

  it('round-trips the alpha plane exactly', () => {
    const source = cutout(48)
    const decoded = decode(encode(source, { lossless: false, quality: 80 }))

    // The colour is lossy; the mask is not.
    for (let i = 3; i < source.data.length; i += 4)
      expect(decoded.data[i]).toBe(source.data[i])
  })

  it('keeps the plain VP8 form when every pixel is opaque', () => {
    const opaque = cutout(16)
    for (let i = 3; i < opaque.data.length; i += 4) opaque.data[i] = 255
    expect(parseRiff(encode(opaque, { lossless: false })).map(chunk => chunk.fourCC)).toEqual(['VP8 '])
  })

  it('drops alpha only when the caller says the image has none', () => {
    const encoded = encode({ ...cutout(16), hasAlpha: false }, { lossless: false })
    expect(parseRiff(encoded).map(chunk => chunk.fourCC)).toEqual(['VP8 '])
  })
})
