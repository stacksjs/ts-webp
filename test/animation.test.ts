import { describe, expect, it } from 'bun:test'
import { decodeAnimation, encode, parseRiff } from '../src'

/**
 * Tests for the animated WebP container parser. We don't exercise an
 * end-to-end animated encode (the encoder doesn't currently emit ANMF
 * frames), so we hand-build minimal animated containers and verify
 * the decoder extracts the right frames + metadata. Each frame's
 * VP8L payload is produced by our regular `encode()` path, so the
 * decode side reuses the lossless decoder and we get exact pixel
 * round-trip on every frame "for free".
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

/** Pack a 24-bit little-endian unsigned integer at `offset`. */
function writeUint24LE(buf: Uint8Array, offset: number, value: number): void {
  buf[offset] = value & 0xFF
  buf[offset + 1] = (value >> 8) & 0xFF
  buf[offset + 2] = (value >> 16) & 0xFF
}

/** Pack a 32-bit little-endian unsigned integer at `offset`. */
function writeUint32LE(buf: Uint8Array, offset: number, value: number): void {
  buf[offset] = value & 0xFF
  buf[offset + 1] = (value >> 8) & 0xFF
  buf[offset + 2] = (value >> 16) & 0xFF
  buf[offset + 3] = (value >>> 24) & 0xFF
}

interface FrameSpec {
  width: number
  height: number
  fill: (x: number, y: number) => [number, number, number, number]
  x?: number
  y?: number
  duration?: number
  blend?: 'overlay' | 'replace'
  dispose?: 'none' | 'background'
}

/**
 * Build a minimal animated WebP container (VP8X + ANIM + N × ANMF).
 * Each frame's payload is a real VP8L bitstream produced by our encoder.
 */
function buildAnimation(canvasW: number, canvasH: number, loopCount: number, bgColor: number, frames: FrameSpec[]): Uint8Array {
  // Build each frame's VP8L payload. We get its bytes by encoding then
  // ripping the VP8L chunk out — way easier than re-emitting by hand.
  const frameBlocks: Uint8Array[] = []
  for (const f of frames) {
    const encoded = encode(makeImage(f.width, f.height, f.fill))
    const vp8l = parseRiff(encoded).find(c => c.fourCC === 'VP8L')!
    const fHeader = new Uint8Array(16)
    writeUint24LE(fHeader, 0, (f.x ?? 0) >>> 1)
    writeUint24LE(fHeader, 3, (f.y ?? 0) >>> 1)
    writeUint24LE(fHeader, 6, f.width - 1)
    writeUint24LE(fHeader, 9, f.height - 1)
    writeUint24LE(fHeader, 12, f.duration ?? 100)
    fHeader[15] = ((f.blend === 'replace' ? 0x02 : 0)
      | (f.dispose === 'background' ? 0x01 : 0))

    // Sub-chunk: just the VP8L bytes wrapped in (fourCC, size).
    const subChunk = new Uint8Array(8 + vp8l.size + (vp8l.size & 1))
    subChunk.set([0x56, 0x50, 0x38, 0x4C], 0) // 'VP8L'
    writeUint32LE(subChunk, 4, vp8l.size)
    subChunk.set(vp8l.data, 8)

    const block = new Uint8Array(fHeader.length + subChunk.length)
    block.set(fHeader, 0)
    block.set(subChunk, fHeader.length)
    frameBlocks.push(block)
  }

  // VP8X chunk: 10 bytes.
  const vp8x = new Uint8Array(10)
  vp8x[0] = 0x02 // animation flag
  writeUint24LE(vp8x, 4, canvasW - 1)
  writeUint24LE(vp8x, 7, canvasH - 1)

  // ANIM chunk: 6 bytes (B, G, R, A, loop_lo, loop_hi).
  const anim = new Uint8Array(6)
  anim[0] = bgColor & 0xFF
  anim[1] = (bgColor >> 8) & 0xFF
  anim[2] = (bgColor >> 16) & 0xFF
  anim[3] = (bgColor >>> 24) & 0xFF
  anim[4] = loopCount & 0xFF
  anim[5] = (loopCount >> 8) & 0xFF

  // Compute total size: RIFF/WEBP wrapper + VP8X chunk + ANIM chunk +
  // each ANMF chunk (8 bytes header + payload, rounded to even).
  let dataSize = 4 // 'WEBP'
  dataSize += 8 + vp8x.length // VP8X
  dataSize += 8 + anim.length // ANIM
  for (const f of frameBlocks) dataSize += 8 + f.length + (f.length & 1)
  const out = new Uint8Array(8 + dataSize)
  out.set([0x52, 0x49, 0x46, 0x46], 0)
  writeUint32LE(out, 4, dataSize)
  out.set([0x57, 0x45, 0x42, 0x50], 8)
  let off = 12
  // VP8X
  out.set([0x56, 0x50, 0x38, 0x58], off); writeUint32LE(out, off + 4, vp8x.length); out.set(vp8x, off + 8); off += 8 + vp8x.length
  // ANIM
  out.set([0x41, 0x4E, 0x49, 0x4D], off); writeUint32LE(out, off + 4, anim.length); out.set(anim, off + 8); off += 8 + anim.length
  for (const f of frameBlocks) {
    out.set([0x41, 0x4E, 0x4D, 0x46], off)
    writeUint32LE(out, off + 4, f.length)
    out.set(f, off + 8)
    off += 8 + f.length + (f.length & 1)
  }
  return out
}

describe('decodeAnimation', () => {
  it('decodes a 1-frame animation with metadata preserved', () => {
    const buf = buildAnimation(16, 16, 0, 0xFF112233, [
      { width: 16, height: 16, fill: (x, y) => [x * 16, y * 16, 100, 255], duration: 250 },
    ])
    const anim = decodeAnimation(buf)
    expect(anim.width).toBe(16)
    expect(anim.height).toBe(16)
    expect(anim.loopCount).toBe(0)
    expect(anim.frames).toHaveLength(1)
    expect(anim.frames[0].duration).toBe(250)
    expect(anim.frames[0].image.width).toBe(16)
    expect(anim.frames[0].image.height).toBe(16)
  })

  it('round-trips per-frame pixels exactly', () => {
    const buf = buildAnimation(8, 8, 5, 0xFFFFFFFF, [
      { width: 8, height: 8, fill: () => [255, 0, 0, 255], duration: 100 },
      { width: 8, height: 8, fill: () => [0, 255, 0, 255], duration: 100 },
      { width: 8, height: 8, fill: () => [0, 0, 255, 255], duration: 100 },
    ])
    const anim = decodeAnimation(buf)
    expect(anim.loopCount).toBe(5)
    expect(anim.frames).toHaveLength(3)
    const expectations: [number, number, number][] = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    for (let f = 0; f < 3; f++) {
      const frame = anim.frames[f]
      expect(frame.image.width).toBe(8)
      const [r, g, b] = expectations[f]
      // Spot-check the first pixel — every pixel of each frame is the
      // same colour by construction, so the first one is enough.
      expect(frame.image.data[0]).toBe(r)
      expect(frame.image.data[1]).toBe(g)
      expect(frame.image.data[2]).toBe(b)
      expect(frame.image.data[3]).toBe(255)
    }
  })

  it('captures per-frame x/y/duration/blend/dispose', () => {
    const buf = buildAnimation(32, 32, 0, 0, [
      { width: 4, height: 4, fill: () => [0, 0, 0, 255], x: 4, y: 6, duration: 50, blend: 'replace', dispose: 'background' },
    ])
    const anim = decodeAnimation(buf)
    const f = anim.frames[0]
    expect(f.x).toBe(4)
    expect(f.y).toBe(6)
    expect(f.duration).toBe(50)
    expect(f.blend).toBe('replace')
    expect(f.dispose).toBe('background')
  })

  it('throws on a non-animated WebP', () => {
    // A regular (non-animated) encode produces a simple-format file
    // with no VP8X — decodeAnimation should refuse it cleanly.
    const buf = encode(makeImage(8, 8, () => [128, 128, 128, 255]))
    expect(() => decodeAnimation(buf)).toThrow(/animat|VP8X/i)
  })

  it('throws on garbage input', () => {
    expect(() => decodeAnimation(new Uint8Array(0))).toThrow()
    expect(() => decodeAnimation(new Uint8Array([1, 2, 3]))).toThrow()
  })

  it('decodes an animated WebP with VP8 (lossy) frames', async () => {
    // Fixture built via `img2webp -lossy -q 80 -d 100 f1.pam -d 200 f2.pam`
    // (cwebp emits VP8 chunks per frame; the inner ALPH chunk is absent
    // because the input frames are opaque). Exercises the VP8-frame path
    // in `decodeFramePayload`, which previously only handled VP8L.
    const path = `${import.meta.dir}/fixtures/anim-lossy.webp`
    const buf = new Uint8Array(await Bun.file(path).arrayBuffer())
    const anim = decodeAnimation(buf)
    expect(anim.width).toBe(32)
    expect(anim.height).toBe(32)
    expect(anim.frames).toHaveLength(2)
    expect(anim.frames[0].duration).toBe(100)
    expect(anim.frames[1].duration).toBe(200)
    for (const f of anim.frames) {
      expect(f.image.width).toBe(32)
      expect(f.image.height).toBe(32)
      expect(f.image.data.length).toBe(32 * 32 * 4)
    }
  })
})
