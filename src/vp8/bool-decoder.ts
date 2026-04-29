/**
 * VP8 boolean (arithmetic) decoder.
 *
 * VP8 partition data is encoded with a binary arithmetic coder where
 * each bit comes with a *probability* that lets it pack roughly
 * `-log2(p)` bits of entropy into the wire. Decoding is a tight loop
 * over a `(range, value)` state pair: each `readBit(prob)` call decides
 * which subrange of `[0, range)` the value falls in, returns the bit,
 * and renormalises so range stays ≥ 128.
 *
 * Used by `vp8/decoder.ts` to read the frame header, segmentation
 * map, mode-info, and DCT coefficient bits.
 *
 * Reference: RFC 6386 (VP8 Data Format and Decoding Guide), §7.
 */

export class BoolDecoder {
  private readonly data: Uint8Array
  private pos: number
  /** Current arithmetic-coder range, always in [128, 255]. */
  private range: number
  /** Current 24-bit value buffer. Refilled byte-by-byte from `data`. */
  private value: number
  /** Bits remaining in `value` before we need to pull another byte. */
  private bitsRemaining: number

  constructor(data: Uint8Array, offset = 0) {
    this.data = data
    this.pos = offset
    this.range = 255
    this.value = 0
    this.bitsRemaining = 0
    // Initialise with two bytes shifted into `value`. RFC 6386 §7.3
    // describes this exact priming.
    if (this.pos < data.length) {
      this.value = data[this.pos++] << 8
    }
    if (this.pos < data.length) {
      this.value |= data[this.pos++]
    }
    this.bitsRemaining = 0
  }

  /**
   * Decode a single bit with the given probability (0..255). The bit
   * value is what the encoder originally inserted; the probability
   * affects the entropy cost on the wire, not the bit's identity.
   */
  readBit(prob: number): number {
    // `split` partitions the current range. Bits with `value < split`
    // get assigned 0; bits ≥ split get 1.
    const split = 1 + (((this.range - 1) * prob) >> 8)
    const bigSplit = split << 8
    let bit: number
    if (this.value >= bigSplit) {
      this.range = this.range - split
      this.value = this.value - bigSplit
      bit = 1
    } else {
      this.range = split
      bit = 0
    }
    // Renormalise: shift range left until it's ≥ 128, pulling more
    // input bits into `value` to keep precision.
    while (this.range < 128) {
      this.range <<= 1
      this.value <<= 1
      this.bitsRemaining++
      if (this.bitsRemaining === 8) {
        this.bitsRemaining = 0
        if (this.pos < this.data.length) {
          this.value |= this.data[this.pos++]
        }
      }
    }
    return bit
  }

  /** Decode a flat (non-arithmetic) `bits`-wide unsigned integer. */
  readLiteral(bits: number): number {
    let v = 0
    for (let i = bits - 1; i >= 0; i--) {
      v |= this.readBit(128) << i
    }
    return v
  }

  /** Decode a signed integer: `bits`-wide magnitude + 1-bit sign. */
  readSignedLiteral(bits: number): number {
    const magnitude = this.readLiteral(bits)
    const sign = this.readBit(128)
    return sign ? -magnitude : magnitude
  }

  /** Byte position in the input stream — useful for splitting partitions. */
  position(): number {
    return this.pos
  }
}
