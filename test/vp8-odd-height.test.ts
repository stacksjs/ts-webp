import { expect, test } from 'bun:test'
import { decode, encode } from '../src/index'

// The final chroma row exists for an odd height. Dropping it left the last
// two luma rows as transparent black, despite VP8 having no alpha plane.
test('decodes every row of an opaque lossy WebP with an odd height', () => {
  for (const height of [1, 3, 5, 15, 17, 31, 33]) {
    const width = 19
    const data = new Uint8Array(width * height * 4)
    for (let i = 0; i < data.length; i += 4) data.set([200, 40, 40, 255], i)
    const decoded = decode(encode({ data, width, height }, { lossless: false, quality: 95 }))
    expect([decoded.width, decoded.height]).toEqual([width, height])
    for (let i = 0; i < decoded.data.length; i += 4) {
      expect(decoded.data[i + 3], `${width}x${height} pixel ${i / 4}`).toBe(255)
      expect(Math.abs(decoded.data[i]! - 200)).toBeLessThanOrEqual(3)
    }
  }
})
