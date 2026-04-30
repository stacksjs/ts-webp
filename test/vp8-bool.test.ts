import { describe, expect, it } from 'bun:test'
import { BoolDecoder } from '../src/vp8/bool-decoder'
import { BoolEncoder } from '../src/vp8/bool-encoder'

/**
 * Round-trip tests for the VP8 boolean (arithmetic) codec. The
 * encoder/decoder pair is the foundation of every VP8 lossy bitstream
 * we emit, so these tests exercise the carry path in detail — a single
 * bit lost or shifted there corrupts every subsequent symbol.
 */

describe('BoolEncoder ⇆ BoolDecoder', () => {
  it('round-trips a flat 1024-bit sequence at p=128', () => {
    const enc = new BoolEncoder()
    const bits: number[] = []
    let seed = 12345
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
      return seed
    }
    for (let i = 0; i < 1024; i++) {
      const b = rand() & 1
      bits.push(b)
      enc.writeBit(0x80, b)
    }
    const buf = enc.finish()
    const dec = new BoolDecoder(buf)
    for (let i = 0; i < 1024; i++) {
      expect(dec.readBit(0x80)).toBe(bits[i])
    }
  })

  it('round-trips with skewed probabilities (255, 1, 200, 50)', () => {
    const probs = [255, 1, 200, 50]
    for (const p of probs) {
      const enc = new BoolEncoder()
      const bits: number[] = []
      let seed = 7777
      const rand = () => {
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return seed
      }
      for (let i = 0; i < 512; i++) {
        const b = rand() & 1
        bits.push(b)
        enc.writeBit(p, b)
      }
      const buf = enc.finish()
      const dec = new BoolDecoder(buf)
      for (let i = 0; i < 512; i++) {
        expect(dec.readBit(p)).toBe(bits[i])
      }
    }
  })

  it('round-trips literals and signed literals', () => {
    const enc = new BoolEncoder()
    enc.writeLiteral(0xAB, 8)
    enc.writeLiteral(0x1234, 16)
    enc.writeLiteral(0, 7)
    enc.writeLiteral(127, 7)
    enc.writeSignedLiteral(-12, 5)
    enc.writeSignedLiteral(7, 5)
    const buf = enc.finish()
    const dec = new BoolDecoder(buf)
    expect(dec.readLiteral(8)).toBe(0xAB)
    expect(dec.readLiteral(16)).toBe(0x1234)
    expect(dec.readLiteral(7)).toBe(0)
    expect(dec.readLiteral(7)).toBe(127)
    expect(dec.readSignedLiteral(5)).toBe(-12)
    expect(dec.readSignedLiteral(5)).toBe(7)
  })

  it('handles long all-zero / all-one runs (carry stress)', () => {
    const enc = new BoolEncoder()
    for (let i = 0; i < 200; i++) enc.writeBit(0x01, 1) // forces lots of carry-prone state
    for (let i = 0; i < 200; i++) enc.writeBit(0xFE, 0)
    const buf = enc.finish()
    const dec = new BoolDecoder(buf)
    for (let i = 0; i < 200; i++) expect(dec.readBit(0x01)).toBe(1)
    for (let i = 0; i < 200; i++) expect(dec.readBit(0xFE)).toBe(0)
  })
})
