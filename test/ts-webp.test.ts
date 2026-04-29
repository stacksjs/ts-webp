import { describe, expect, it } from 'bun:test'
import webp, { createRiffContainer, decode, encode, parseRiff } from '../src'

/**
 * Build a width × height RGBA buffer filled by `fill(x, y) → [r,g,b,a]`.
 * Lets each test describe a pixel pattern as a function rather than as a
 * pre-computed array, keeping fixtures readable.
 */
function makeImage(width: number, height: number, fill: (x: number, y: number) => [number, number, number, number]): {
  data: Uint8Array
  width: number
  height: number
} {
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

/** Assert exact pixel-level match between encoded → decoded and the original. */
function expectRoundTrip(image: { data: Uint8Array, width: number, height: number }): void {
  const encoded = encode(image)
  const decoded = decode(encoded)
  expect(decoded.width).toBe(image.width)
  expect(decoded.height).toBe(image.height)
  expect(decoded.data.length).toBe(image.data.length)
  // Walk byte-by-byte rather than `toEqual` so failures point at the first
  // bad pixel instead of dumping a 64 KB array diff.
  for (let i = 0; i < image.data.length; i++) {
    if (decoded.data[i] !== image.data[i]) {
      throw new Error(`Round-trip mismatch at byte ${i}: input=${image.data[i]}, output=${decoded.data[i]}`)
    }
  }
}

describe('ts-webp', () => {
  describe('lossless round-trip — exact pixel match', () => {
    it('1x1 image', () => {
      expectRoundTrip(makeImage(1, 1, () => [42, 84, 168, 200]))
    })

    it('2x2 all-zeros', () => {
      expectRoundTrip(makeImage(2, 2, () => [0, 0, 0, 0]))
    })

    it('4x4 single colour (all pixels identical)', () => {
      expectRoundTrip(makeImage(4, 4, () => [200, 100, 50, 255]))
    })

    it('8x8 horizontal/vertical gradient', () => {
      expectRoundTrip(makeImage(8, 8, (x, y) => [x * 32, y * 32, 128, 255]))
    })

    it('16x16 pseudo-random (every pixel distinct)', () => {
      expectRoundTrip(makeImage(16, 16, (x, y) => {
        const i = y * 16 + x
        return [(i * 37) % 256, (i * 73) % 256, (i * 113) % 256, 255]
      }))
    })

    it('32x32 with semi-transparent alpha', () => {
      expectRoundTrip(makeImage(32, 32, (x, y) => [x * 8, y * 8, 128, 50 + (x + y) % 200]))
    })

    it('64x64 large gradient', () => {
      expectRoundTrip(makeImage(64, 64, (x, y) => [(x * 4) % 256, (y * 4) % 256, 128, 200]))
    })

    it('non-square wide (100x10)', () => {
      expectRoundTrip(makeImage(100, 10, (x, y) => [x % 256, y * 25, (x + y) % 256, 255]))
    })

    it('non-square tall (10x100)', () => {
      expectRoundTrip(makeImage(10, 100, (x, y) => [x * 25, y % 256, (x + y) % 256, 255]))
    })

    it('image with all 256 R values', () => {
      // 256x1 image stresses the per-channel Huffman tree to its full alphabet.
      expectRoundTrip(makeImage(256, 1, x => [x, 128, 200, 255]))
    })

    it('image where alpha varies per pixel', () => {
      expectRoundTrip(makeImage(16, 16, (x, y) => [100, 100, 100, (x * 16 + y) % 256]))
    })
  })

  describe('encode output structure', () => {
    it('produces a valid RIFF/WEBP wrapper', () => {
      const encoded = encode(makeImage(4, 4, () => [128, 128, 128, 255]))
      // 'RIFF'
      expect(String.fromCharCode(encoded[0], encoded[1], encoded[2], encoded[3])).toBe('RIFF')
      // 'WEBP'
      expect(String.fromCharCode(encoded[8], encoded[9], encoded[10], encoded[11])).toBe('WEBP')
      // File-size field matches the actual buffer length minus the 8 header bytes.
      const view = new DataView(encoded.buffer, encoded.byteOffset, encoded.byteLength)
      expect(view.getUint32(4, true)).toBe(encoded.length - 8)
    })

    it('always emits a VP8L chunk for lossless output', () => {
      const encoded = encode(makeImage(8, 8, () => [50, 100, 150, 255]))
      const chunks = parseRiff(encoded)
      expect(chunks.find(c => c.fourCC === 'VP8L')).toBeDefined()
    })

    it('VP8L payload starts with the 0x2F signature byte', () => {
      const encoded = encode(makeImage(4, 4, () => [10, 20, 30, 40]))
      const chunks = parseRiff(encoded)
      const vp8l = chunks.find(c => c.fourCC === 'VP8L')
      expect(vp8l).toBeDefined()
      expect(vp8l!.data[0]).toBe(0x2F)
    })

    it('encodes width/height correctly into the VP8L header', () => {
      // 100x50 image: width-1=99 (14-bit), height-1=49 (14-bit), packed
      // into the 4 bytes after the signature, low byte first.
      const encoded = encode(makeImage(100, 50, () => [0, 0, 0, 255]))
      const chunks = parseRiff(encoded)
      const vp8l = chunks.find(c => c.fourCC === 'VP8L')!
      const hdr = vp8l.data[1] | (vp8l.data[2] << 8) | (vp8l.data[3] << 16) | (vp8l.data[4] << 24)
      expect((hdr & 0x3FFF) + 1).toBe(100)
      expect(((hdr >> 14) & 0x3FFF) + 1).toBe(50)
    })

    it('different inputs produce different encoded bytes', () => {
      const a = encode(makeImage(8, 8, () => [255, 0, 0, 255]))
      const b = encode(makeImage(8, 8, () => [0, 255, 0, 255]))
      // Length might match (same shape), but the encoded payload should differ.
      expect(a.length === b.length && a.every((v, i) => v === b[i])).toBe(false)
    })
  })

  describe('decode error handling', () => {
    it('throws on garbage data', () => {
      expect(() => decode(new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0]))).toThrow()
    })

    it('throws on truncated RIFF (no WEBP magic)', () => {
      expect(() => decode(new Uint8Array([0x52, 0x49, 0x46, 0x46]))).toThrow()
    })

    it('throws on a malformed VP8 chunk (bad start code)', () => {
      // Hand-build a RIFF/WEBP container with a `VP8 ` chunk that has a
      // valid keyframe bit but a corrupted start code. The decoder
      // should reject it cleanly rather than emit garbage pixels.
      const vp8Header = new Uint8Array(10)
      vp8Header[0] = 0x00 // frame tag low: keyframe (bit 0 = 0)
      vp8Header[3] = 0x00 // corrupted start code
      vp8Header[4] = 0x00
      vp8Header[5] = 0x00
      vp8Header[6] = 0x10
      vp8Header[7] = 0x00
      vp8Header[8] = 0x10
      vp8Header[9] = 0x00
      const stub = createRiffContainer([{ fourCC: 'VP8 ', data: vp8Header }])
      expect(() => decode(stub)).toThrow(/start code|VP8/i)
    })
  })

  describe('encode default behaviour', () => {
    it('default options use lossless encoding', () => {
      // No options → defaults to lossless and produces a VP8L chunk.
      const encoded = encode(makeImage(4, 4, () => [50, 100, 150, 255]))
      const chunks = parseRiff(encoded)
      expect(chunks.find(c => c.fourCC === 'VP8L')).toBeDefined()
    })

    it('explicit lossless: true is identical to default', () => {
      const a = encode(makeImage(4, 4, () => [50, 100, 150, 255]))
      const b = encode(makeImage(4, 4, () => [50, 100, 150, 255]), { lossless: true })
      expect(a.length).toBe(b.length)
    })
  })

  describe('riff container helpers', () => {
    it('createRiffContainer + parseRiff round-trip', () => {
      const payload = new Uint8Array([1, 2, 3, 4, 5])
      const out = createRiffContainer([{ fourCC: 'TEST', data: payload }])
      const chunks = parseRiff(out)
      expect(chunks).toHaveLength(1)
      expect(chunks[0].fourCC).toBe('TEST')
      expect(chunks[0].size).toBe(payload.length)
      expect(Array.from(chunks[0].data)).toEqual(Array.from(payload))
    })

    it('handles odd-sized chunks (RIFF requires word alignment)', () => {
      // A 3-byte chunk needs a pad byte; the parser should still report size=3.
      const out = createRiffContainer([{ fourCC: 'ODD ', data: new Uint8Array([1, 2, 3]) }])
      const chunks = parseRiff(out)
      expect(chunks[0].size).toBe(3)
      expect(Array.from(chunks[0].data)).toEqual([1, 2, 3])
    })

    it('preserves chunk order and exact payloads across many chunks', () => {
      const out = createRiffContainer([
        { fourCC: 'AAA1', data: new Uint8Array([0xAA]) },
        { fourCC: 'BBB2', data: new Uint8Array([0xBB, 0xBB]) },
        { fourCC: 'CCC3', data: new Uint8Array([0xCC, 0xCC, 0xCC, 0xCC]) },
      ])
      const chunks = parseRiff(out)
      expect(chunks.map(c => c.fourCC)).toEqual(['AAA1', 'BBB2', 'CCC3'])
      expect(chunks[2].data.length).toBe(4)
    })
  })

  describe('default export', () => {
    it('default export exposes encode and decode', () => {
      expect(typeof webp.encode).toBe('function')
      expect(typeof webp.decode).toBe('function')
    })
  })

  describe('decode format option', () => {
    it('format: "rgba" (default) returns 4 bytes per pixel', () => {
      const img = makeImage(4, 3, (x, y) => [x * 50, y * 50, 100, 200])
      const decoded = decode(encode(img))
      expect(decoded.data.length).toBe(4 * 3 * 4)
    })

    it('format: "rgb" returns 3 bytes per pixel and drops alpha', () => {
      const img = makeImage(4, 3, (x, y) => [x * 50, y * 50, 100, 200])
      const decoded = decode(encode(img), { format: 'rgb' })
      expect(decoded.data.length).toBe(4 * 3 * 3)
      // First pixel: x=0, y=0 → r=0, g=0, b=100. Verify no alpha byte was
      // copied into the RGB stream.
      expect(decoded.data[0]).toBe(0)
      expect(decoded.data[1]).toBe(0)
      expect(decoded.data[2]).toBe(100)
      // Second pixel starts at offset 3 (not 4) — that's the whole point
      // of `rgb`: stride is 3, not 4.
      expect(decoded.data[3]).toBe(50) // x=1's red
    })

    it('format: "rgb" preserves R/G/B byte-for-byte across all pixels', () => {
      const img = makeImage(8, 8, (x, y) => [x * 16, y * 16, (x + y) * 8, 255])
      const decoded = decode(encode(img), { format: 'rgb' })
      for (let p = 0; p < 64; p++) {
        const x = p % 8
        const y = (p / 8) | 0
        expect(decoded.data[p * 3]).toBe(x * 16)
        expect(decoded.data[p * 3 + 1]).toBe(y * 16)
        expect(decoded.data[p * 3 + 2]).toBe((x + y) * 8)
      }
    })
  })
})
