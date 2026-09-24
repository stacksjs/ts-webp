import { describe, expect, it } from 'bun:test'
import { decode, encode } from '../src/index'

/**
 * A backreference that reaches far back survives the round trip.
 *
 * The encoder packed each prefix-coded value into a 32-bit token with 16 bits
 * left for the extra value. Distance codes 36-39 need 17 or 18, so a match
 * more than about 262k pixels back was truncated: a valid bitstream that
 * decoded to the wrong pixels from there on. Every existing fixture was too
 * small to produce one. A 560x1024 alpha mask did, and came out as streaks.
 */
describe('VP8L long-distance backreferences', () => {
  it('round-trips a match more than 2^18 pixels back', () => {
    const width = 600
    const height = 600
    const data = new Uint8Array(width * height * 4)

    // Seeded noise so nothing matches locally, then the last ten rows copy
    // the first ten: the only matches are ~354,000 pixels back.
    let seed = 0x2545F491
    const next = () => {
      seed ^= seed << 13
      seed ^= seed >>> 17
      seed ^= seed << 5
      return seed >>> 0
    }
    for (let i = 0; i < width * height; i++) {
      const value = next()
      data[i * 4] = value & 0xFF
      data[i * 4 + 1] = (value >>> 8) & 0xFF
      data[i * 4 + 2] = (value >>> 16) & 0xFF
      data[i * 4 + 3] = 255
    }
    const rowBytes = width * 4
    data.copyWithin((height - 10) * rowBytes, 0, 10 * rowBytes)

    const decoded = decode(encode({ data, width, height }, { lossless: true }))

    let mismatches = 0
    for (let i = 0; i < data.length; i++) {
      if (decoded.data[i] !== data[i])
        mismatches++
    }
    expect(mismatches).toBe(0)
  })
})
