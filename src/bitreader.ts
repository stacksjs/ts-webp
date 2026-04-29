/**
 * Bit-level I/O for VP8L bitstreams.
 *
 * Both reader and writer pack bits LSB-first inside each byte, matching the
 * VP8L specification. Internally they keep a 32-bit accumulator so most bit
 * operations are simple shifts and masks — the per-bit shift+push pattern
 * the previous implementation used was easy to read but ~10× slower on
 * realistic-sized images.
 */

/**
 * Streaming bit-level reader. The accumulator pulls bytes lazily, so reading
 * past the end produces zeros (rather than throwing) — which matches what
 * libwebp does and keeps decoders robust against bitstreams that are
 * implicitly zero-padded at the end.
 */
export class BitReader {
  private readonly data: Uint8Array
  /** Index of the next byte to pull into the accumulator. */
  private pos = 0
  /** Low `accBits` bits of `acc` are unread bits, ready to consume. */
  private acc = 0
  private accBits = 0
  /** Position of the *next* bit to read, in bits from the start of `data`. */
  private bitPos = 0

  constructor(data: Uint8Array) {
    this.data = data
  }

  /** Pull bytes into the accumulator until it has ≥ 25 bits buffered. */
  private fill(): void {
    while (this.accBits <= 24 && this.pos < this.data.length) {
      this.acc |= this.data[this.pos++] << this.accBits
      this.accBits += 8
    }
  }

  /** Look at the next `n` bits (1 ≤ n ≤ 24) without advancing the cursor. */
  peek(n: number): number {
    if (this.accBits < n) this.fill()
    return this.acc & ((1 << n) - 1)
  }

  /** Advance the cursor by `n` bits. Cheap — no byte work, just shifts. */
  consume(n: number): void {
    this.acc >>>= n
    this.accBits -= n
    this.bitPos += n
  }

  /** Read a single bit and return 0 or 1. */
  readBit(): number {
    if (this.accBits < 1) this.fill()
    const bit = this.acc & 1
    this.acc >>>= 1
    this.accBits -= 1
    this.bitPos += 1
    return bit
  }

  /** Read `n` bits (1 ≤ n ≤ 24). LSB-first within each byte. */
  readBits(n: number): number {
    if (n === 0) return 0
    if (n > 24) throw new Error(`readBits: max 24 bits at a time, got ${n}`)
    if (this.accBits < n) this.fill()
    const value = this.acc & ((1 << n) - 1)
    this.acc >>>= n
    this.accBits -= n
    this.bitPos += n
    return value
  }

  /** Skip ahead to the next byte boundary. */
  alignToByte(): void {
    const drop = this.bitPos & 7
    if (drop) this.consume(8 - drop)
  }

  /** Bits read so far (counting from the start of `data`). */
  getBitPosition(): number {
    return this.bitPos
  }

  /** Byte index that contains the next bit. */
  getBytePosition(): number {
    return this.bitPos >> 3
  }

  /** Whether there are more bits available — including buffered ones. */
  hasMore(): boolean {
    return this.accBits > 0 || this.pos < this.data.length
  }
}

/**
 * Streaming bit-level writer with auto-growing buffer. Like the reader, it
 * keeps a 32-bit accumulator and only commits whole bytes — a `writeBits(x,
 * 8)` is one mask + one assignment, not eight push() calls.
 */
export class BitWriter {
  private buffer: Uint8Array
  private byteLen = 0
  private acc = 0
  private accBits = 0

  constructor(initialCapacity = 1024) {
    this.buffer = new Uint8Array(initialCapacity)
  }

  private ensureCapacity(extra: number): void {
    const needed = this.byteLen + extra
    if (needed <= this.buffer.length) return
    let cap = this.buffer.length || 1
    while (cap < needed) cap *= 2
    const grown = new Uint8Array(cap)
    grown.set(this.buffer.subarray(0, this.byteLen))
    this.buffer = grown
  }

  /** Write a single bit (LSB of `bit`). */
  writeBit(bit: number): void {
    this.acc |= (bit & 1) << this.accBits
    this.accBits++
    if (this.accBits >= 8) this.flushBytes()
  }

  /** Write the low `n` bits of `value`. 1 ≤ n ≤ 24. */
  writeBits(value: number, n: number): void {
    if (n === 0) return
    if (n > 24) throw new Error(`writeBits: max 24 bits at a time, got ${n}`)
    this.acc |= (value & ((1 << n) - 1)) << this.accBits
    this.accBits += n
    if (this.accBits >= 8) this.flushBytes()
  }

  private flushBytes(): void {
    while (this.accBits >= 8) {
      this.ensureCapacity(1)
      this.buffer[this.byteLen++] = this.acc & 0xFF
      this.acc >>>= 8
      this.accBits -= 8
    }
  }

  /**
   * Pad the current byte with zeros so the next write starts on a byte
   * boundary. Used at the end of a chunk before writing untranslated bytes.
   */
  alignToByte(): void {
    if (this.accBits === 0) return
    this.ensureCapacity(1)
    this.buffer[this.byteLen++] = this.acc & 0xFF
    this.acc = 0
    this.accBits = 0
  }

  /** Flush the partial byte (if any) and return the buffer as a Uint8Array. */
  getBuffer(): Uint8Array {
    this.alignToByte()
    return this.buffer.slice(0, this.byteLen)
  }

  /** Bytes that would be in the buffer right now (for size estimation). */
  byteLength(): number {
    return this.byteLen + (this.accBits > 0 ? 1 : 0)
  }
}
