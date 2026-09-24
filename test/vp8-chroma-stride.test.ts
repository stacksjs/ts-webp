import { describe, expect, it } from 'bun:test'
import { decode, encode } from '../src/index'

/**
 * Chroma survives lossy encoding.
 *
 * The chroma forward DCT read its 8-wide residual with the luma stride of 16,
 * so the bottom half of every 8x8 chroma block was transformed from zeros and
 * kept only its prediction. Saturated colours drifted from block to block and
 * photos grew horizontal colour bands, and raising the quality did nothing -
 * q92 was as wrong as q80. Neutral greys were unaffected, which is how it hid.
 */
function image(width: number, height: number, pixel: (x: number, y: number) => [number, number, number]) {
  const data = new Uint8Array(width * height * 4)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b] = pixel(x, y)
      const i = (y * width + x) * 4
      data[i] = r
      data[i + 1] = g
      data[i + 2] = b
      data[i + 3] = 255
    }
  }
  return { data, width, height }
}

function meanError(a: Uint8Array, b: Uint8Array): number {
  let total = 0
  let n = 0
  for (let i = 0; i < a.length; i += 4) {
    total += Math.abs(a[i]! - b[i]!) + Math.abs(a[i + 1]! - b[i + 1]!) + Math.abs(a[i + 2]! - b[i + 2]!)
    n += 3
  }
  return total / n
}

describe('VP8 chroma', () => {
  it('holds a flat saturated colour in every macroblock at high quality', () => {
    for (const colour of [[200, 40, 40], [40, 200, 40], [40, 40, 200], [230, 160, 110]] as const) {
      const source = image(64, 64, () => [...colour])
      const decoded = decode(encode(source, { lossless: false, quality: 95 }))
      let worst = 0
      for (let i = 0; i < source.data.length; i += 4) {
        for (let c = 0; c < 3; c++)
          worst = Math.max(worst, Math.abs(decoded.data[i + c]! - source.data[i + c]!))
      }
      expect(worst).toBeLessThanOrEqual(2)
    }
  })

  it('gets closer to the source as quality rises', () => {
    const source = image(96, 96, (x, y) => [(x * 5 + y) & 255, (y * 3) & 255, (x * 2 + y * 2) & 255])
    const low = meanError(decode(encode(source, { lossless: false, quality: 40 })).data, source.data)
    const high = meanError(decode(encode(source, { lossless: false, quality: 95 })).data, source.data)
    expect(high).toBeLessThan(low)
  })
})
