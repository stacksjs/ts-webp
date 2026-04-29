/**
 * VP8 boolean (arithmetic) decoder.
 *
 * 1:1 port of libwebp's `VP8BitReader` (see `VP8GetBit` in
 * `src/utils/bit_reader_inl_utils.h`). We implement the simpler
 * `7 ^ BitsLog2Floor(range)` renormalisation formula instead of the
 * table-driven `VP8GetBitAlt` variant — both produce identical state.
 *
 * State layout:
 *   range  — true range (in [128, 256] post-renormalisation), stored
 *            via `br->range = range - 1`.
 *   value  — the live arithmetic value. The current "decision byte"
 *            sits at bit positions `bits..bits+7`; we shift the value
 *            down by `bits` to read those bits.
 *   bits   — count of bits left in `value` before we need to refill.
 *            Starts at -8 so the constructor's first `loadFinalBytes`
 *            triggers a load.
 *
 * `value` can grow to ~40 bits during shifts — we keep it in a JS
 * Number and use multiplication / `Math.floor(value / 2^pos)` so we
 * don't get caught by JS's signed 32-bit bitwise ops.
 */

function bitsLog2Floor(n: number): number {
  // floor(log2(n)). n must be ≥ 1.
  let log = 0
  while (n >= 2) {
    n >>>= 1
    log++
  }
  return log
}

export class BoolDecoder {
  private readonly buf: Uint8Array
  private bufPos: number
  private readonly bufEnd: number
  /** Stored as `actual_range - 1` (libwebp convention). */
  private rangeM1: number
  /** Decoder value register; can extend up to ~40 bits during shifts. */
  private value: number
  /** Bits remaining before refill (libwebp's `bits`, signed). */
  private bitsLeft: number
  private eof: boolean

  constructor(data: Uint8Array, offset = 0, length?: number) {
    this.buf = data
    this.bufPos = offset
    this.bufEnd = length === undefined ? data.length : Math.min(offset + length, data.length)
    this.rangeM1 = 254
    this.value = 0
    this.bitsLeft = -8
    this.eof = false
    this.loadFinalBytes()
  }

  /**
   * Refill `value` with one more byte of input. We use the
   * load-one-byte-at-a-time variant; libwebp's "fast" path loads 4
   * bytes at once but produces the same final state.
   */
  private loadFinalBytes(): void {
    if (this.bufPos < this.bufEnd) {
      this.bitsLeft += 8
      this.value = this.buf[this.bufPos++] + this.value * 256
    }
    else if (!this.eof) {
      this.value *= 256
      this.bitsLeft += 8
      this.eof = true
    }
    else {
      this.bitsLeft = 0
    }
  }

  /**
   * Decode a single bit with the given probability (0..255). `prob` is
   * the probability the encoder thought a 0 would appear, expressed as
   * a fraction over 256.
   */
  readBit(prob: number): number {
    if (this.bitsLeft < 0) this.loadFinalBytes()

    // libwebp uses `range = stored + 1` (the true range) for the split
    // computation; the post-decision range is then stored back as
    // `actual_range - 1`.
    const range = this.rangeM1 + 1
    const pos = this.bitsLeft
    const split = ((range - 1) * prob) >>> 8
    const value = Math.floor(this.value / 2 ** pos) // hi byte of value
    let newRange: number
    let bit: number
    if (value > split) {
      newRange = range - (split + 1)
      this.value -= (split + 1) * 2 ** pos
      bit = 1
    }
    else {
      newRange = split + 1
      bit = 0
    }
    // Renormalise so `range` ≥ 128.
    if (newRange < 128) {
      const shift = 7 - bitsLog2Floor(newRange)
      newRange <<= shift
      this.bitsLeft -= shift
    }
    this.rangeM1 = newRange - 1
    return bit
  }

  /** Decode a flat (uniform-probability) `bits`-wide unsigned integer. */
  readLiteral(bits: number): number {
    let v = 0
    while (bits-- > 0) {
      v |= this.readBit(0x80) << bits
    }
    return v
  }

  /** Decode a signed integer: `bits`-wide magnitude + 1-bit sign. */
  readSignedLiteral(bits: number): number {
    const value = this.readLiteral(bits)
    return this.readBit(0x80) ? -value : value
  }

  /** Byte position in the input stream — useful for splitting partitions. */
  position(): number {
    return this.bufPos
  }

  /** True if the decoder has run out of input (libwebp's `eof`). */
  isEof(): boolean {
    return this.eof
  }
}
