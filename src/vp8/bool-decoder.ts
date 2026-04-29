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
 * We mirror libvpx's 32-bit decoder register: 4 bytes are pre-loaded
 * into `value` at construction, and the comparison against `bigSplit =
 * split << 24` operates on the high 8 bits of the 32-bit register. A
 * `count` field tracks how many bits remain in the loaded value before
 * we need to pull another byte from the input stream.
 *
 * Reference: RFC 6386 (VP8 Data Format and Decoding Guide), §7;
 * libvpx `vp8/decoder/dboolhuff.{c,h}`.
 */

export class BoolDecoder {
  private readonly data: Uint8Array
  private pos: number
  /** Last valid byte index + 1 — the bool decoder won't read past this. */
  private readonly end: number
  /** Current arithmetic-coder range, always in [128, 255] after renormalisation. */
  private range: number
  /** 32-bit value register; the top 8 bits drive every decision. */
  private value: number
  /**
   * Bits remaining in `value` before we need to pull another byte.
   * Mirrors libvpx's `count`: starts at -8, increments by `shift` per
   * readBit, refills when it goes ≥ 0.
   */
  private count: number

  constructor(data: Uint8Array, offset = 0, length?: number) {
    this.data = data
    this.pos = offset
    this.end = length === undefined ? data.length : Math.min(offset + length, data.length)
    this.range = 255
    this.value = 0
    this.count = -8
    // Pre-load 4 bytes (32 bits) into `value`, packing the first byte
    // at the top of the register so the high byte drives the first
    // decision. If the input runs out, missing bytes are zero-padded
    // — which leaves `value` smaller and naturally biases subsequent
    // reads toward bit=0, matching the RFC 6386 §7.3 expectation that
    // an exhausted stream produces 0 bits indefinitely.
    //
    // We use plain multiplication rather than `<<` because JS bitwise
    // ops force a 32-bit signed integer and would produce a negative
    // number once a byte's high bit reaches position 31.
    for (let i = 0; i < 4; i++) {
      this.value = this.value * 256
      if (this.pos < this.end) {
        this.value += this.data[this.pos++]
      }
    }
  }

  /**
   * Decode a single bit with the given probability (0..255). The bit
   * value is what the encoder originally inserted; `prob` is the chance
   * the encoder thought a 0 would appear, expressed as a fraction over 256.
   */
  readBit(prob: number): number {
    const split = 1 + (((this.range - 1) * prob) >> 8)
    // bigSplit = split << 24 — using *0x1000000 because `<< 24` produces
    // negative numbers in JS when the high bit of `split` is set.
    const bigSplit = split * 0x1000000
    let bit: number
    if (this.value >= bigSplit) {
      this.range = this.range - split
      this.value = this.value - bigSplit
      bit = 1
    }
    else {
      this.range = split
      bit = 0
    }
    // Renormalise: shift range left until it's ≥ 128, pulling more
    // input bits into `value` to keep precision.
    while (this.range < 128) {
      this.range <<= 1
      this.value *= 2
      this.count++
      if (this.count === 0) {
        this.count = -8
        if (this.pos < this.end) {
          this.value += this.data[this.pos++]
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
