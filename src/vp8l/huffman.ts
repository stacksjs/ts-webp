import type { BitReader, BitWriter } from '../bitreader'

/**
 * Canonical Huffman code construction + serialization for VP8L.
 *
 * "Canonical" means the (length, symbol) pair fully determines the code:
 * codes are assigned in lex order of (length, symbol) starting at 0, with
 * the bit ordering fixed by the spec. Both encoder and decoder agree on
 * the exact code without ever transmitting raw codes — only the per-symbol
 * lengths.
 *
 * The decoder side uses a small primary LUT (≤ 8 bits) that resolves any
 * code that fits, falling back to a tree walk for longer codes. For
 * realistic inputs the LUT hits ≥ 95 % of symbols, so decode is essentially
 * O(1) per symbol.
 *
 * Reference: WebP Lossless Bitstream Specification §7.2.
 */

/** Order of code-length codes in the bitstream (per VP8L spec §7.2.2). */
const CODE_LENGTH_CODE_ORDER = new Int32Array([
  17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
])
const NUM_CODE_LENGTH_CODES = 19

/** Maximum code length VP8L allows for any tree (per spec). */
const MAX_CODE_LENGTH = 15

// ---------------------------------------------------------------------------
// Encode-side: build code lengths from frequencies
// ---------------------------------------------------------------------------

/**
 * Build canonical Huffman code lengths from per-symbol frequencies, with
 * lengths capped at `maxBits`. Uses package-merge to honour the cap; the
 * cap matters because VP8L disallows codes longer than 15 bits.
 *
 * Returns a `Uint8Array` of length `freq.length` where `lengths[s]` is the
 * number of bits assigned to symbol `s`, or 0 if `freq[s] === 0`.
 */
export function buildCodeLengths(freq: Uint32Array | number[], maxBits = MAX_CODE_LENGTH): Uint8Array {
  const n = freq.length
  const lengths = new Uint8Array(n)

  // Collect non-zero symbols.
  const symbols: { sym: number, w: number }[] = []
  for (let s = 0; s < n; s++) {
    const w = (freq as ArrayLike<number>)[s]
    if (w > 0) symbols.push({ sym: s, w })
  }

  if (symbols.length === 0) return lengths
  // A single-symbol code still gets length 1 — VP8L's "simple code" path can
  // store this even more efficiently, but a length-1 normal tree also works.
  if (symbols.length === 1) {
    lengths[symbols[0].sym] = 1
    return lengths
  }

  // Package-merge gives the optimal length-limited prefix code. The simpler
  // "build full Huffman tree, then if any depth > maxBits fall back to flat"
  // approach can produce sub-optimal codes for skewed distributions; our
  // tests check round-trip but not output size, so technically we could be
  // lazy here — but package-merge is short and gets us materially better
  // compression on real images at no real cost.
  symbols.sort((a, b) => a.w - b.w || a.sym - b.sym)

  const m = symbols.length
  // Each "package" is (weight, set-of-symbols-it-contributes-to). We
  // represent the set as a packed bitfield since m ≤ 280 in our usage and
  // a pair of `BigInt`s would dominate cost. Instead, track per-symbol
  // counters: `count[s]` is how many "leaves" of weight `freq[s]` are
  // packed in across rounds. Length of `s` = `count[s]`.
  const count = new Uint32Array(m)

  // Round k: merge sorted packages from round k+1 plus the original leaves
  // pairwise; each merge increments `count` for every original leaf in the
  // package. We do this lazily by tracking each package as the multiset of
  // its leaves; cheap enough at our scale.
  type Pkg = { w: number, leaves: Uint16Array }
  let prev: Pkg[] = symbols.map((s, i) => ({
    w: s.w,
    leaves: new Uint16Array([i]),
  }))

  for (let k = 0; k < maxBits; k++) {
    // Pair up `prev` into packages of two.
    const paired: Pkg[] = []
    for (let i = 0; i + 1 < prev.length; i += 2) {
      const a = prev[i]
      const b = prev[i + 1]
      const merged = new Uint16Array(a.leaves.length + b.leaves.length)
      merged.set(a.leaves, 0)
      merged.set(b.leaves, a.leaves.length)
      paired.push({ w: a.w + b.w, leaves: merged })
    }
    // Merge with the original leaves and re-sort.
    const next: Pkg[] = paired.slice()
    for (let i = 0; i < m; i++) {
      next.push({ w: symbols[i].w, leaves: new Uint16Array([i]) })
    }
    next.sort((x, y) => x.w - y.w)
    prev = next
  }

  // Take the smallest 2m − 2 packages from the final round; each leaf of
  // each chosen package contributes 1 to `count[leaf]`.
  const take = 2 * m - 2
  for (let i = 0; i < take && i < prev.length; i++) {
    const leaves = prev[i].leaves
    for (let j = 0; j < leaves.length; j++) count[leaves[j]]++
  }

  // Map back to original symbols.
  for (let i = 0; i < m; i++) lengths[symbols[i].sym] = Math.min(count[i] || 1, maxBits)

  return lengths
}

/**
 * Convert per-symbol lengths into the canonical code value for each symbol.
 *
 * Returns a `Uint32Array` of codes; bit ordering matches what VP8L expects:
 * bits are emitted high-to-low, which on the wire (LSB-first within byte)
 * means we *must* reverse before writing.
 */
export function lengthsToCodes(lengths: Uint8Array): Uint32Array {
  const n = lengths.length
  const codes = new Uint32Array(n)

  let maxLen = 0
  for (let i = 0; i < n; i++) if (lengths[i] > maxLen) maxLen = lengths[i]
  if (maxLen === 0) return codes

  const blCount = new Uint32Array(maxLen + 1)
  for (let i = 0; i < n; i++) if (lengths[i] > 0) blCount[lengths[i]]++

  const nextCode = new Uint32Array(maxLen + 1)
  let code = 0
  for (let bits = 1; bits <= maxLen; bits++) {
    code = (code + blCount[bits - 1]) << 1
    nextCode[bits] = code
  }

  for (let s = 0; s < n; s++) {
    const len = lengths[s]
    if (len > 0) codes[s] = nextCode[len]++
  }

  return codes
}

/** Reverse the low `bits` bits of `value`. Used to convert canonical codes to wire order. */
export function reverseBits(value: number, bits: number): number {
  let r = 0
  for (let i = 0; i < bits; i++) r = (r << 1) | ((value >> i) & 1)
  return r
}

// ---------------------------------------------------------------------------
// Encode-side: serialize a Huffman tree into the bitstream
// ---------------------------------------------------------------------------

/**
 * Write a Huffman tree to the bitstream using VP8L's "normal code" path —
 * which itself uses a tiny meta-Huffman code over the 19 code-length-codes.
 *
 * VP8L also has a "simple code" fast-path for trees with 1 or 2 non-zero
 * lengths; we use it when applicable since it's tiny on the wire.
 */
export function writeHuffmanTree(writer: BitWriter, lengths: Uint8Array): void {
  // Count non-zero symbols.
  let nonZero = 0
  let firstSym = -1
  let secondSym = -1
  for (let s = 0; s < lengths.length; s++) {
    if (lengths[s] > 0) {
      nonZero++
      if (firstSym < 0) firstSym = s
      else if (secondSym < 0) secondSym = s
      if (nonZero > 2) break
    }
  }

  // ── Simple code path ──
  // VP8L's "simple code" can encode trees with 1 or 2 non-zero symbols
  // when each fits in 8 bits. For longer alphabets (the green/length tree
  // includes length codes up to 279) we fall through to the normal path.
  // is_first_8_bits semantics: 1 → 8-bit symbol, 0 → 1-bit symbol (so the
  // 1-bit form only works when sym ∈ {0, 1}).
  if (nonZero === 1 && firstSym < 256) {
    writer.writeBit(1) // simple-code flag
    writer.writeBit(0) // num_symbols - 1 = 0
    if (firstSym <= 1) {
      writer.writeBit(0) // is_first_8_bits = 0 → 1-bit symbol
      writer.writeBits(firstSym, 1)
    } else {
      writer.writeBit(1) // is_first_8_bits = 1 → 8-bit symbol
      writer.writeBits(firstSym, 8)
    }
    return
  }
  if (nonZero === 2 && firstSym < 256 && secondSym < 256) {
    writer.writeBit(1) // simple-code flag
    writer.writeBit(1) // num_symbols - 1 = 1
    if (firstSym <= 1) {
      writer.writeBit(0) // is_first_8_bits = 0 → 1-bit symbol1
      writer.writeBits(firstSym, 1)
    } else {
      writer.writeBit(1) // is_first_8_bits = 1 → 8-bit symbol1
      writer.writeBits(firstSym, 8)
    }
    writer.writeBits(secondSym, 8) // symbol2 is always 8-bit
    return
  }

  // ── Normal code path ──
  writer.writeBit(0) // simple-code flag = 0

  // Build code-length frequencies for symbols 0..15 only — runs (16/17/18)
  // are an *output* of the encoder of code lengths, not an input, and are
  // chosen by the run-length compressor below. Using the literal lengths
  // for the meta-tree is what every reference implementation does.
  const clFreq = new Uint32Array(NUM_CODE_LENGTH_CODES)
  for (let s = 0; s < lengths.length; s++) clFreq[lengths[s]]++

  // Build lengths for the meta-tree, capped at 7 bits per VP8L.
  const clLengths = buildCodeLengths(clFreq, 7)
  const clCodes = lengthsToCodes(clLengths)

  // Number of code-length-codes to send: enough to cover the highest-index
  // non-zero entry in CODE_LENGTH_CODE_ORDER. VP8L always sends at least 4.
  let numCodes = NUM_CODE_LENGTH_CODES
  while (numCodes > 4 && clLengths[CODE_LENGTH_CODE_ORDER[numCodes - 1]] === 0) numCodes--

  writer.writeBits(numCodes - 4, 4)
  for (let i = 0; i < numCodes; i++) {
    writer.writeBits(clLengths[CODE_LENGTH_CODE_ORDER[i]], 3)
  }

  // "Use length"? If 0, the decoder reads code lengths until it has filled
  // every alphabet entry. If 1, an explicit length prefix says how many
  // code-lengths to read. We always send 0 — simpler, and only marginally
  // bigger on the wire.
  writer.writeBit(0)

  // Emit the symbol-length sequence using meta-tree codes (literal 0..15)
  // plus run codes (16/17/18). For now skip the run-length compression and
  // emit each length literally — saves complexity at the cost of a few
  // bytes per tree on sparse alphabets.
  for (let s = 0; s < lengths.length; s++) {
    const len = lengths[s]
    const codeLen = clLengths[len]
    const code = clCodes[len]
    writer.writeBits(reverseBits(code, codeLen), codeLen)
  }
}

// ---------------------------------------------------------------------------
// Decode-side: read trees + decode symbols
// ---------------------------------------------------------------------------

/**
 * A built Huffman tree ready to read symbols from a bitstream.
 *
 * Internally a primary LUT for codes ≤ `lutBits` resolves the common case
 * in O(1); longer codes use a packed binary tree (`Int16Array` parent →
 * (left, right) child indices) walked bit-by-bit.
 */
export class HuffmanTree {
  private lut!: Int32Array // (codeLen << 16) | symbol, or -1 if no entry
  private lutBits!: number
  /** Tree storage: node `i` lives at `tree[i*2]` (left), `tree[i*2+1]` (right). Leaves encode `-(symbol+1)`. */
  private tree!: Int32Array
  private treeNext = 0
  /** `true` once buildFromLengths has completed; cheap guard against decoding from a half-built tree. */
  private built = false

  /**
   * Build the tree from a length-per-symbol array. Lengths follow VP8L's
   * canonical assignment: codes are sorted by `(length, symbol)` and given
   * sequential values starting at 0.
   */
  buildFromLengths(lengths: Uint8Array | number[]): void {
    const n = lengths.length
    let maxLen = 0
    let nonZero = 0
    for (let i = 0; i < n; i++) {
      const len = (lengths as ArrayLike<number>)[i]
      if (len > 0) {
        nonZero++
        if (len > maxLen) maxLen = len
      }
    }

    // LUT covers the first `lutBits` bits of every code. Pick the smaller
    // of 8 and `maxLen`: a smaller LUT is faster to build and just as fast
    // to read; an 8-bit LUT for a code with maxLen=4 wastes 240 entries.
    this.lutBits = Math.min(8, Math.max(1, maxLen))
    this.lut = new Int32Array(1 << this.lutBits).fill(-1)
    // Tree size bound: complete binary tree of depth maxLen has at most
    // 2 * 2^maxLen nodes, but a Huffman tree with N leaves has 2N-1 nodes
    // total. Use the tighter bound to avoid allocating ~64KB for small
    // alphabets.
    this.tree = new Int32Array(2 * Math.max(2 * nonZero, 2))
    this.treeNext = 1 // node 0 is the root

    if (nonZero === 0) {
      this.built = true
      return
    }

    if (nonZero === 1) {
      // Single-symbol tree: the encoder still emits one canonical bit
      // (length-1 code value 0) so the bit stream has a well-defined
      // length. The decoder must consume that bit to keep its cursor
      // aligned, even though there's only one possible symbol.
      let onlySym = 0
      let onlyLen = 1
      for (let i = 0; i < n; i++) {
        const len = (lengths as ArrayLike<number>)[i]
        if (len > 0) { onlySym = i; onlyLen = len; break }
      }
      // LUT covers `lutBits` peeked bits; for any peek, return `(onlyLen, onlySym)`
      // so readSymbol consumes `onlyLen` bits and returns the single symbol.
      for (let i = 0; i < this.lut.length; i++) this.lut[i] = (onlyLen << 16) | onlySym
      this.built = true
      return
    }

    // Build canonical codes.
    const codes = lengthsToCodes(lengths instanceof Uint8Array ? lengths : new Uint8Array(lengths))

    // Insert each (length, symbol) into both LUT and tree.
    for (let s = 0; s < n; s++) {
      const len = (lengths as ArrayLike<number>)[s]
      if (len === 0) continue
      const code = codes[s]
      if (len <= this.lutBits) {
        // Replicate across LUT entries that share these `len` low bits.
        const reversed = reverseBits(code, len)
        const fill = 1 << (this.lutBits - len)
        for (let i = 0; i < fill; i++) {
          this.lut[reversed | (i << len)] = (len << 16) | s
        }
      } else {
        // Walk into the tree, allocating nodes as needed.
        let node = 0
        for (let bit = len - 1; bit >= 0; bit--) {
          const direction = (code >> bit) & 1
          const slot = node * 2 + direction
          if (bit === 0) {
            this.tree[slot] = -(s + 1)
          } else {
            if (this.tree[slot] === 0) {
              if (this.treeNext * 2 + 1 >= this.tree.length) {
                const grown = new Int32Array(this.tree.length * 2)
                grown.set(this.tree)
                this.tree = grown
              }
              this.tree[slot] = this.treeNext++
            }
            node = this.tree[slot]
          }
        }
      }
    }

    this.built = true
  }

  /** Read one symbol. Fast path = LUT, fallback = tree walk for long codes. */
  readSymbol(reader: BitReader): number {
    if (!this.built) throw new Error('HuffmanTree.readSymbol called before buildFromLengths')
    const peek = reader.peek(this.lutBits)
    const entry = this.lut[peek]
    if (entry >= 0) {
      const len = entry >>> 16
      reader.consume(len)
      return entry & 0xFFFF
    }
    // LUT miss → walk the tree from the root, consuming one bit at a time.
    // peek() didn't advance the cursor, so the tree walk re-reads the bits
    // we used as the LUT key — exactly what's wanted, since the tree path
    // encodes the *full* canonical code from MSB to LSB.
    let node = 0
    while (true) {
      const direction = reader.readBit()
      const child = this.tree[node * 2 + direction]
      if (child < 0) return -(child + 1)
      if (child === 0) throw new Error('Invalid Huffman code: walked into uninitialised node')
      node = child
    }
  }
}

// ---------------------------------------------------------------------------
// Decode-side: read the tree itself from the bitstream
// ---------------------------------------------------------------------------

/**
 * Read a Huffman tree from `reader` for an alphabet of `numSymbols` symbols.
 * Mirrors `writeHuffmanTree` exactly.
 */
export function readHuffmanTree(reader: BitReader, numSymbols: number): HuffmanTree {
  const tree = new HuffmanTree()
  const isSimple = reader.readBit()

  if (isSimple) {
    const numSyms = reader.readBit() + 1
    const isFirst8 = reader.readBit()
    const sym1 = reader.readBits(isFirst8 ? 8 : 1)
    if (numSyms === 1) {
      const lengths = new Uint8Array(numSymbols)
      if (sym1 < numSymbols) lengths[sym1] = 1
      tree.buildFromLengths(lengths)
    } else {
      const sym2 = reader.readBits(8) // symbol2 is always 8-bit
      const lengths = new Uint8Array(numSymbols)
      if (sym1 < numSymbols) lengths[sym1] = 1
      if (sym2 < numSymbols) lengths[sym2] = 1
      tree.buildFromLengths(lengths)
    }
    return tree
  }

  // Normal code.
  const numCodes = 4 + reader.readBits(4)
  const clLengths = new Uint8Array(NUM_CODE_LENGTH_CODES)
  for (let i = 0; i < numCodes; i++) {
    clLengths[CODE_LENGTH_CODE_ORDER[i]] = reader.readBits(3)
  }
  const clTree = new HuffmanTree()
  clTree.buildFromLengths(clLengths)

  const useLength = reader.readBit()
  let maxSymbol = numSymbols
  if (useLength) {
    const lengthNbits = 2 + 2 * reader.readBits(3)
    maxSymbol = 2 + reader.readBits(lengthNbits)
  }

  const lengths = new Uint8Array(numSymbols)
  let prevLen = 8
  let count = 0
  let i = 0
  while (i < numSymbols && count < maxSymbol) {
    const code = clTree.readSymbol(reader)
    if (code < 16) {
      lengths[i++] = code
      if (code !== 0) prevLen = code
      count++
    } else if (code === 16) {
      const repeat = 3 + reader.readBits(2)
      for (let j = 0; j < repeat && i < numSymbols; j++) lengths[i++] = prevLen
      count += repeat
    } else if (code === 17) {
      const repeat = 3 + reader.readBits(3)
      for (let j = 0; j < repeat && i < numSymbols; j++) lengths[i++] = 0
      count += repeat
    } else { // code === 18
      const repeat = 11 + reader.readBits(7)
      for (let j = 0; j < repeat && i < numSymbols; j++) lengths[i++] = 0
      count += repeat
    }
  }

  tree.buildFromLengths(lengths)
  return tree
}
