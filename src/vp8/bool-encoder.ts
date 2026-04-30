/**
 * VP8 boolean (arithmetic) encoder — direct port of libwebp's
 * `VP8PutBit` (`utils/bit_writer_utils.c`). Pairs with `bool-decoder.ts`
 * (which is a port of libwebp's `VP8GetBit`) so that any encoder output
 * is bit-compatible with `dwebp` and our own decoder.
 *
 * State invariant matches libwebp's `VP8BitWriter`:
 *   • `range` — actual range (1..255), starts at 255
 *   • `value` — running register holding the high bits we'll flush
 *   • `nb_bits` — bits in `value` we haven't flushed yet (starts at -8)
 *   • `run` — count of pending 0xFF bytes; carry-aware delayed flush
 *
 * The carry handling is delicate: while flushing, if the next byte we
 * want to emit is 0xFF, we *queue* it (incrementing `run`) instead of
 * emitting. If a later byte forces a carry (the `0x100` overflow bit),
 * those queued 0xFFs become 0x00 and the byte before them gets +1.
 *
 * Reference: libwebp `src/utils/bit_writer_utils.c`. Anchored to the
 * `kNorm` / `kNewRange` lookup tables and the `Flush` carry algorithm.
 */

// Direct copy of libwebp's `kNorm[128]` from `bit_writer_utils.c`. Each
// entry is `7 - log2_floor(stored_range)`; gives the renormalisation
// shift count to apply when `range < 127`.
const VP8_NORM = new Uint8Array([
  7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
])

const VP8_NEW_RANGE = new Uint8Array([
  127, 127, 191, 127, 159, 191, 223, 127, 143, 159, 175, 191, 207, 223, 239,
  127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239,
  247, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179,
  183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239,
  243, 247, 251, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149,
  151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179,
  181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209,
  211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239,
  241, 243, 245, 247, 249, 251, 253, 127,
])

export class BoolEncoder {
  private buf: Uint8Array
  private pos = 0
  /** Running arithmetic-coding register. Up to ~32 bits before flush. */
  private value = 0
  /** Active range — STORED form (= actual - 1), libwebp convention.
   * The decoder's `BoolDecoder` also stores `actual - 1`; using the
   * same convention here keeps the `(range * prob) >> 8` formula
   * symmetric across encode and decode. Starts at 254 (= 255 - 1). */
  private range = 254
  /** Bits queued for flush; libwebp starts at -8. */
  private nbBits = -8
  /** Pending 0xFF byte count (delayed flush for carry handling). */
  private run = 0

  constructor(initialCapacity = 1024) {
    this.buf = new Uint8Array(initialCapacity)
  }

  private grow(extra: number): void {
    while (this.pos + extra > this.buf.length) {
      const next = new Uint8Array(this.buf.length * 2)
      next.set(this.buf)
      this.buf = next
    }
  }

  private flush(): void {
    const s = 8 + this.nbBits
    // Use multiplication to safely shift `value` up to 32 bits.
    const bits = Math.floor(this.value / (2 ** s))
    this.value -= bits * (2 ** s)
    this.nbBits -= 8
    if ((bits & 0xFF) !== 0xFF) {
      // Reserve enough room for run + 1 bytes.
      this.grow(this.run + 1)
      if (bits & 0x100) {
        // Carry: bump the previous byte by 1, walk back over the 0xFF
        // run flipping them to 0x00.
        if (this.pos > 0) this.buf[this.pos - 1] = (this.buf[this.pos - 1] + 1) & 0xFF
      }
      const fillValue = (bits & 0x100) ? 0x00 : 0xFF
      while (this.run > 0) {
        this.buf[this.pos++] = fillValue
        this.run--
      }
      this.buf[this.pos++] = bits & 0xFF
    }
    else {
      // Defer 0xFF byte emission until we know whether a carry will rip
      // through it.
      this.run++
    }
  }

  /**
   * Write one bit with the given probability of a zero (0..255).
   * Mirrors libwebp's `VP8PutBit` exactly.
   */
  writeBit(prob: number, bit: number): void {
    const split = (this.range * prob) >>> 8
    if (bit) {
      this.value += split + 1
      this.range -= split + 1
    }
    else {
      this.range = split
    }
    if (this.range < 127) {
      const shift = VP8_NORM[this.range]
      this.range = VP8_NEW_RANGE[this.range]
      // Shift value left by `shift` — multiply for >32-bit safety.
      this.value = this.value * (2 ** shift)
      this.nbBits += shift
      if (this.nbBits > 0) this.flush()
    }
  }

  /**
   * Write a flat (uniform-probability) `bits`-wide unsigned integer,
   * MSB first via `writeBitUniform`. Matches libwebp's `VP8PutBits`.
   */
  writeLiteral(value: number, bits: number): void {
    for (let mask = 1 << (bits - 1); mask > 0; mask >>>= 1) {
      this.writeBitUniform(value & mask ? 1 : 0)
    }
  }

  /**
   * Write a uniform sign bit. Matches libwebp's `VP8PutBitUniform`,
   * which uses a halved `split` (= range >> 1) and a 1-bit shift.
   */
  writeBitUniform(bit: number): void {
    const split = this.range >>> 1
    if (bit) {
      this.value += split + 1
      this.range -= split + 1
    }
    else {
      this.range = split
    }
    if (this.range < 127) {
      this.range = VP8_NEW_RANGE[this.range]
      this.value = this.value * 2
      this.nbBits += 1
      if (this.nbBits > 0) this.flush()
    }
  }

  /**
   * Write a signed literal. Mirrors libwebp's `VP8PutSignedBits`:
   * a uniform 1-bit "is zero" flag, then for non-zero values a
   * `(magnitude << 1) | sign_bit` value with `bits + 1` width.
   *
   * Decoder counterpart `readSignedLiteral` in `bool-decoder.ts` reads
   * this back via `readLiteral(bits) ⊕ readBit(0x80) ? -value : value`,
   * so we serialise to match that.
   */
  writeSignedLiteral(value: number, bits: number): void {
    const mag = value < 0 ? -value : value
    this.writeLiteral(mag, bits)
    this.writeBit(0x80, value < 0 ? 1 : 0)
  }

  /**
   * Finalise and return the encoded bytes. Mirrors libwebp's
   * `VP8BitWriterFinish`: pad with `9 - nb_bits` zero bits via
   * uniform writes (so every queued bit gets pushed out), then do one
   * final flush + drain any remaining pending 0xFF run.
   */
  finish(): Uint8Array {
    const padBits = 9 - this.nbBits
    for (let i = 0; i < padBits; i++) this.writeBitUniform(0)
    this.nbBits = 0
    this.flush()
    if (this.run > 0) {
      this.grow(this.run)
      while (this.run > 0) {
        this.buf[this.pos++] = 0xFF
        this.run--
      }
    }
    return this.buf.subarray(0, this.pos)
  }

  byteLength(): number {
    return this.pos
  }
}
