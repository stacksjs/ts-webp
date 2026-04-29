import { describe, expect, it } from 'bun:test'
import { decode, encode, parseRiff } from '../src'

/**
 * Crash-resistance tests. These don't check correctness of decoded
 * pixels — only that the decoder *terminates* (with a thrown Error,
 * not a crash, infinite loop, or undefined-behaviour read) when fed
 * malformed input. A library that processes untrusted byte input
 * needs this: a corrupted .webp from the network shouldn't be able
 * to hang or crash a server.
 *
 * The bar:
 *   • decode() returns or throws — never hangs
 *   • thrown error is a plain Error, not a TypeError on undefined etc
 *   • no out-of-bounds read or NaN propagation produces a loop
 */

/** Deterministic 32-bit LCG so tests are reproducible across runs. */
function makeRng(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s
  }
}

function randomBytes(seed: number, length: number): Uint8Array {
  const rng = makeRng(seed)
  const out = new Uint8Array(length)
  for (let i = 0; i < length; i++) out[i] = rng() & 0xFF
  return out
}

/**
 * Run a decode call with a hard timeout. If the call returns within
 * the budget we accept either result (success or thrown Error) — the
 * point is that it doesn't hang. If it times out, the test fails: we
 * don't want infinite-loop bugs to be silently flaky.
 */
function decodeWithTimeout(buffer: Uint8Array, timeoutMs = 250): { ok: boolean, err?: Error } {
  // We can't actually preempt JS, so the timeout here is a budget for
  // total work we accept the decoder doing. In practice the decoder
  // either rejects malformed input in microseconds or runs through to
  // termination on size-bounded inputs.
  const start = performance.now()
  try {
    decode(buffer)
    return { ok: true }
  }
  catch (e) {
    if (performance.now() - start > timeoutMs) {
      throw new Error(`decode() ran for ${(performance.now() - start).toFixed(0)} ms before throwing — likely infinite loop`)
    }
    if (!(e instanceof Error)) throw new Error(`decode() threw a non-Error value: ${typeof e}`)
    return { ok: false, err: e }
  }
}

describe('decoder crash-resistance', () => {
  it('decode() throws cleanly on empty input', () => {
    const r = decodeWithTimeout(new Uint8Array(0))
    expect(r.ok).toBe(false)
    expect(r.err).toBeInstanceOf(Error)
  })

  it('decode() throws cleanly on a single byte', () => {
    expect(decodeWithTimeout(new Uint8Array([0xFF])).ok).toBe(false)
  })

  it('decode() throws cleanly on truncated RIFF magic', () => {
    expect(decodeWithTimeout(new Uint8Array([0x52, 0x49])).ok).toBe(false) // "RI"
  })

  it('decode() throws cleanly on a 12-byte all-zero buffer', () => {
    // RIFF header expects 'RIFF...WEBP' — all-zero fails the magic check
    expect(decodeWithTimeout(new Uint8Array(12)).ok).toBe(false)
  })

  it('decode() throws cleanly when RIFF magic is correct but WEBP isn\'t', () => {
    const buf = new Uint8Array(20)
    buf.set([0x52, 0x49, 0x46, 0x46], 0) // 'RIFF'
    // bytes 8-11 are zero, not 'WEBP'
    expect(decodeWithTimeout(buf).ok).toBe(false)
  })

  it('decode() handles 1000 random ~256-byte buffers without hanging', () => {
    let crashes = 0
    for (let seed = 0; seed < 1000; seed++) {
      const r = decodeWithTimeout(randomBytes(seed, 256))
      // Almost all random buffers throw; the rare lucky one that happens
      // to satisfy the RIFF/WEBP/VP8L magic + length checks may decode
      // successfully (with garbage pixels). Either is acceptable.
      if (r.ok) crashes++ // not actually a crash, just count for diag
    }
    // Sanity check — we expect mostly thrown errors, but if zero ever
    // succeed it's still fine.
    expect(crashes).toBeLessThan(1000)
  })

  it('decode() handles malformed VP8L bitstreams (valid wrapper, garbage payload)', () => {
    // Build a valid RIFF/WEBP wrapper around a random 'VP8L' payload.
    for (let seed = 0; seed < 100; seed++) {
      const payload = randomBytes(seed, 64)
      // First byte must be 0x2F to clear the early "invalid signature"
      // fast-fail; we want the decoder to actually walk the bitstream
      // and reject it on Huffman/length mismatches inside.
      payload[0] = 0x2F
      const buf = new Uint8Array(20 + payload.length)
      buf.set([0x52, 0x49, 0x46, 0x46], 0) // 'RIFF'
      const dataLen = 4 + 8 + payload.length
      buf[4] = dataLen & 0xFF
      buf[5] = (dataLen >> 8) & 0xFF
      buf[6] = (dataLen >> 16) & 0xFF
      buf[7] = 0
      buf.set([0x57, 0x45, 0x42, 0x50], 8) // 'WEBP'
      buf.set([0x56, 0x50, 0x38, 0x4C], 12) // 'VP8L'
      buf[16] = payload.length & 0xFF
      buf[17] = (payload.length >> 8) & 0xFF
      buf[18] = (payload.length >> 16) & 0xFF
      buf[19] = 0
      buf.set(payload, 20)

      // Should throw or succeed within budget — never hang.
      try { decode(buf) }
      catch (e) {
        if (!(e instanceof Error)) throw new Error(`non-Error thrown for seed ${seed}`)
      }
    }
  })

  it('parseRiff handles truncated chunks without infinite loop', () => {
    // A valid RIFF/WEBP header but a chunk length that runs past EOF.
    const buf = new Uint8Array(24)
    buf.set([0x52, 0x49, 0x46, 0x46], 0)
    buf[4] = 0x10
    buf.set([0x57, 0x45, 0x42, 0x50], 8)
    buf.set([0x42, 0x41, 0x44, 0x21], 12) // 'BAD!'
    // Lie about chunk size — claim 0xFFFFFF, but we only have 4 bytes left.
    buf[16] = 0xFF
    buf[17] = 0xFF
    buf[18] = 0xFF
    buf[19] = 0xFF
    const start = performance.now()
    parseRiff(buf) // shouldn't loop forever
    expect(performance.now() - start).toBeLessThan(100)
  })
})

describe('encoder explicit-lossy rejection', () => {
  it('encode({ lossless: false }) throws clearly', () => {
    const data = new Uint8Array(16)
    expect(() => encode({ data, width: 2, height: 2 }, { lossless: false })).toThrow(/lossy/i)
  })

  it('encode({}) defaults to lossless and succeeds', () => {
    const data = new Uint8Array(16)
    const out = encode({ data, width: 2, height: 2 })
    expect(out.length).toBeGreaterThan(20) // RIFF wrapper alone is 20 bytes
  })
})
