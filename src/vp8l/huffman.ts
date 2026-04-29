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
 * lengths capped at `maxBits`. Uses a textbook two-step:
 *
 *   1. Build a *standard* (uncapped) Huffman tree via min-heap merging.
 *      This gives the optimal lengths if no cap were imposed.
 *   2. If any length exceeds `maxBits`, run zlib-style "bit-length
 *      fix-up": iteratively shorten the deepest nodes and lengthen the
 *      shallowest until every length fits and Kraft's inequality holds.
 *
 * Returns a `Uint8Array` of length `freq.length` where `lengths[s]` is the
 * number of bits assigned to symbol `s`, or 0 if `freq[s] === 0`.
 *
 * The earlier package-merge implementation was elegant on paper but
 * subtly miscomputed lengths for sparse distributions in a way that
 * violated Kraft's inequality (sum 2^-len > 1) — and that produced
 * decoder bitstreams the meta-Huffman tree couldn't parse. Standard
 * Huffman + length fix-up is what zlib, deflate, and libwebp use, and
 * it's known correct.
 */
export function buildCodeLengths(freq: Uint32Array | number[], maxBits = MAX_CODE_LENGTH): Uint8Array {
  const n = freq.length
  const lengths = new Uint8Array(n)

  type Node = { weight: number, sym: number, left?: Node, right?: Node }
  const heap: Node[] = []
  for (let s = 0; s < n; s++) {
    const w = (freq as ArrayLike<number>)[s]
    if (w > 0) heap.push({ weight: w, sym: s })
  }

  if (heap.length === 0) return lengths
  if (heap.length === 1) {
    lengths[heap[0].sym] = 1
    return lengths
  }

  // Sort ascending. We use sorted-insert instead of a real heap because n
  // is small (≤ 2328 for VP8L) and `Array.splice` is fine at that scale.
  heap.sort((a, b) => a.weight - b.weight)
  while (heap.length > 1) {
    const a = heap.shift()!
    const b = heap.shift()!
    const m: Node = { weight: a.weight + b.weight, sym: -1, left: a, right: b }
    // Binary-insert m to keep the array sorted.
    let lo = 0
    let hi = heap.length
    while (lo < hi) {
      const mid = (lo + hi) >>> 1
      if (heap[mid].weight <= m.weight) lo = mid + 1
      else hi = mid
    }
    heap.splice(lo, 0, m)
  }

  // Walk the tree to assign per-symbol lengths.
  // Iterative walk to avoid stack overflow on deeply unbalanced trees.
  type Frame = { node: Node, depth: number }
  const stack: Frame[] = [{ node: heap[0], depth: 0 }]
  while (stack.length > 0) {
    const f = stack.pop()!
    if (f.node.sym >= 0) {
      // Single-leaf tree (only one symbol with non-zero freq) gets depth 0
      // from the walk; force it to 1 so we always emit at least one bit.
      lengths[f.node.sym] = Math.max(f.depth, 1)
    } else {
      stack.push({ node: f.node.left!, depth: f.depth + 1 })
      stack.push({ node: f.node.right!, depth: f.depth + 1 })
    }
  }

  // ── Length-limit fix-up ──
  // Find max length; if within the cap, we're done.
  let maxLen = 0
  for (let s = 0; s < n; s++) if (lengths[s] > maxLen) maxLen = lengths[s]
  if (maxLen <= maxBits) return lengths

  // Cap every length at maxBits, then redistribute to restore Kraft.
  // The classic algorithm: count how many entries are at each length,
  // overflow-aware, and shuffle bits up until Kraft's inequality holds
  // (in integer form: ∑ 2^(maxBits - lengths[s]) ≤ 2^maxBits).
  const lenCount = new Uint32Array(maxBits + 2)
  for (let s = 0; s < n; s++) {
    if (lengths[s] > 0) {
      const capped = lengths[s] > maxBits ? maxBits : lengths[s]
      lenCount[capped]++
    }
  }

  // Compute Kraft excess in fixed-point with denominator 2^maxBits.
  // After capping, every code at length L contributes 2^(maxBits - L).
  // The total must equal 2^maxBits exactly for a valid prefix code.
  let kraftSum = 0
  for (let L = 1; L <= maxBits; L++) kraftSum += lenCount[L] * (1 << (maxBits - L))

  const KRAFT_TARGET = 1 << maxBits

  // While the sum overshoots (Kraft > 1 in real form), lengthen something.
  // We lengthen the *shortest* available code (highest contribution to
  // overshoot per unit lengthening), which aligns with the standard fix-up.
  while (kraftSum > KRAFT_TARGET) {
    // Find the smallest length L where we can lengthen one entry.
    // We need an entry at depth L that we can move to L+1 (and L+1 < maxBits
    // so we don't immediately violate the cap again).
    let L = maxBits - 1
    while (L > 0 && lenCount[L] === 0) L--
    if (L === 0) break // Shouldn't happen if input is sane.
    lenCount[L]--
    lenCount[L + 1]++
    kraftSum -= 1 << (maxBits - L - 1) // moving from L to L+1 halves contribution
  }

  // Conversely, the sum could undershoot if we capped a long code — Kraft
  // < 1 means we have spare encoding capacity, which is fine for a prefix
  // code but wastes bits on the wire. Promote the longest entries by 1
  // until we use every bit. (This step is what makes the resulting code
  // *canonical*.)
  while (kraftSum < KRAFT_TARGET) {
    // Find the longest non-empty length and shorten one of its entries.
    let L = maxBits
    while (L > 0 && lenCount[L] === 0) L--
    if (L <= 1) break
    lenCount[L]--
    lenCount[L - 1]++
    kraftSum += 1 << (maxBits - L) // moving from L to L-1 doubles contribution
  }

  // Re-assign per-symbol lengths from the histogram, picking the heaviest
  // symbols to take the shortest codes (canonical assignment).
  const sortedSyms = Array.from({ length: n }, (_, s) => s)
    .filter(s => lengths[s] > 0)
    .sort((a, b) => {
      const fa = (freq as ArrayLike<number>)[a]
      const fb = (freq as ArrayLike<number>)[b]
      return fb - fa // heaviest first
    })

  // Walk lengths from shortest to longest, assigning the count for each.
  let symIdx = 0
  for (let L = 1; L <= maxBits; L++) {
    let remaining = lenCount[L]
    while (remaining > 0 && symIdx < sortedSyms.length) {
      lengths[sortedSyms[symIdx++]] = L
      remaining--
    }
  }
  // Any symbols left after we exhaust lenCount have length 0 — shouldn't
  // happen for non-zero-freq symbols, but defensively zero them out so the
  // canonical-code assignment doesn't see stale data.
  while (symIdx < sortedSyms.length) lengths[sortedSyms[symIdx++]] = 0

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
