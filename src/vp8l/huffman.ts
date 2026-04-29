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

  // Build standard Huffman with a real binary min-heap. The previous
  // implementation used `Array.shift()` + `Array.splice()`, both O(n);
  // that's quadratic in the alphabet size, which is fine for 280-symbol
  // trees but visible at 2328 symbols (the green/length tree's max with
  // 11-bit cache). Real heap is O(n log n).
  //
  // Layout: `nodeWeight`, `nodeSym`, `nodeLeft`, `nodeRight` are parallel
  // arrays indexed by node ID. ID 0 is reserved as "no node" sentinel,
  // which is convenient because uninitialised heap slots are 0. Leaves
  // get `sym ≥ 0`; internal nodes get `sym = -1`.
  const cap = n * 2 + 1 // upper bound: n leaves + (n-1) internal + sentinel
  const nodeWeight = new Uint32Array(cap)
  const nodeSym = new Int32Array(cap)
  const nodeLeft = new Int32Array(cap)
  const nodeRight = new Int32Array(cap)
  let nextNode = 1

  // Min-heap of node IDs ordered by weight; classic binary heap.
  const heap = new Int32Array(n + 1)
  let heapSize = 0

  function heapPush(id: number): void {
    let i = heapSize++
    heap[i] = id
    // Sift up.
    while (i > 0) {
      const parent = (i - 1) >>> 1
      if (nodeWeight[heap[parent]] <= nodeWeight[heap[i]]) break
      const t = heap[parent]; heap[parent] = heap[i]; heap[i] = t
      i = parent
    }
  }

  function heapPop(): number {
    const top = heap[0]
    heap[0] = heap[--heapSize]
    // Sift down.
    let i = 0
    while (true) {
      const l = 2 * i + 1
      const r = 2 * i + 2
      let smallest = i
      if (l < heapSize && nodeWeight[heap[l]] < nodeWeight[heap[smallest]]) smallest = l
      if (r < heapSize && nodeWeight[heap[r]] < nodeWeight[heap[smallest]]) smallest = r
      if (smallest === i) break
      const t = heap[smallest]; heap[smallest] = heap[i]; heap[i] = t
      i = smallest
    }
    return top
  }

  // Seed leaves.
  for (let s = 0; s < n; s++) {
    const w = (freq as ArrayLike<number>)[s]
    if (w > 0) {
      const id = nextNode++
      nodeWeight[id] = w
      nodeSym[id] = s
      heapPush(id)
    }
  }

  if (heapSize === 0) return lengths
  if (heapSize === 1) {
    lengths[nodeSym[heap[0]]] = 1
    return lengths
  }

  // Merge until one node remains.
  while (heapSize > 1) {
    const a = heapPop()
    const b = heapPop()
    const m = nextNode++
    nodeWeight[m] = nodeWeight[a] + nodeWeight[b]
    nodeSym[m] = -1
    nodeLeft[m] = a
    nodeRight[m] = b
    heapPush(m)
  }
  const root = heap[0]

  // Iterative walk to assign per-symbol depths. Stack avoids recursion
  // overflow on pathologically unbalanced trees.
  const walkStack = new Int32Array(cap * 2) // pairs of (node, depth)
  let walkTop = 0
  walkStack[walkTop++] = root
  walkStack[walkTop++] = 0
  while (walkTop > 0) {
    const depth = walkStack[--walkTop]
    const id = walkStack[--walkTop]
    if (nodeSym[id] >= 0) {
      // Single-leaf trees would get depth 0 — force ≥ 1 so we always
      // emit at least one bit per symbol on the wire.
      lengths[nodeSym[id]] = depth > 0 ? depth : 1
    } else {
      walkStack[walkTop++] = nodeLeft[id]
      walkStack[walkTop++] = depth + 1
      walkStack[walkTop++] = nodeRight[id]
      walkStack[walkTop++] = depth + 1
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
 * A built Huffman tree, ready to decode symbols from a bitstream.
 *
 * Storage is a single flat `Int32Array` holding a primary LUT followed
 * by zero or more secondary LUTs ("subtables") — the standard two-level
 * scheme used by every fast inflate decoder. Codes whose length fits in
 * the primary LUT (≤ 8 bits) decode in one peek + consume; longer codes
 * decode via a secondary LUT lookup with no tree walk.
 *
 * Entry encoding (`Int32` per slot):
 *   - Non-negative entry → leaf:
 *       bits 16..23 = code length (1..maxLen)
 *       bits 0..15  = symbol (alphabet is ≤ 2328, fits in 16 bits)
 *   - Negative entry → subtable pointer:
 *       bits 0..23   = offset into `lut` of the subtable's first slot
 *       bits 24..30  = subtable size in bits (so subtable has `1 << subBits` entries)
 *       bit 31       = 1 (the sign bit, used as the discriminator)
 *
 * Subtable leaf entries store the *extra* code length beyond the primary
 * 8 bits in the same `bits 16..23` slot, so the consumer in `readSymbol`
 * adds 8 to that count when computing total bits to consume.
 */
export class HuffmanTree {
  private lut!: Int32Array
  /** Bits consumed by the primary LUT (always 8 unless the alphabet has only short codes). */
  private lutBits = 8
  /** Set once `buildFromLengths` completes; cheap guard against decoding from a half-built tree. */
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
    let firstSym = -1
    let firstLen = 1
    for (let i = 0; i < n; i++) {
      const len = (lengths as ArrayLike<number>)[i]
      if (len > 0) {
        nonZero++
        if (len > maxLen) maxLen = len
        if (firstSym < 0) { firstSym = i; firstLen = len }
      }
    }

    // ── Edge cases ──
    if (nonZero === 0) {
      this.lutBits = 1
      this.lut = new Int32Array(2).fill(-1)
      this.built = true
      return
    }

    if (nonZero === 1) {
      // Single-symbol tree: encoder still emits one canonical bit so the
      // bitstream has a well-defined length. Decoder consumes the bit
      // (its value is irrelevant) and returns the only symbol.
      this.lutBits = 1
      this.lut = new Int32Array(2)
      this.lut[0] = (firstLen << 16) | firstSym
      this.lut[1] = (firstLen << 16) | firstSym
      this.built = true
      return
    }

    // ── Primary LUT size ──
    // For trees whose max code length fits in the primary, use a smaller
    // LUT — saves both build time and per-decode peek cost. For longer
    // trees, always use 8-bit primary; the spec maxes lengths at 15 so
    // the worst-case secondary table has 2^7 = 128 entries.
    this.lutBits = maxLen <= 8 ? Math.max(1, maxLen) : 8
    const primarySize = 1 << this.lutBits

    // ── First pass: figure out how big each secondary table needs to be ──
    // Group long codes by their primary 8-bit prefix; the prefix is the
    // *reversed* code's low `lutBits` bits because that's what the wire
    // peek returns.
    const subBitsByPrefix = new Uint8Array(primarySize)
    if (maxLen > this.lutBits) {
      // Build canonical codes once so we can compute reversed prefixes.
      const codesNorm = lengthsToCodes(lengths instanceof Uint8Array ? lengths : new Uint8Array(lengths))
      for (let s = 0; s < n; s++) {
        const len = (lengths as ArrayLike<number>)[s]
        if (len <= this.lutBits) continue
        const reversed = reverseBits(codesNorm[s], len)
        const prefix = reversed & (primarySize - 1)
        const extra = len - this.lutBits
        if (extra > subBitsByPrefix[prefix]) subBitsByPrefix[prefix] = extra
      }
    }

    // ── Allocate a single flat array for primary + all secondaries ──
    let totalSize = primarySize
    const subOffsetByPrefix = new Int32Array(primarySize).fill(-1)
    for (let p = 0; p < primarySize; p++) {
      if (subBitsByPrefix[p] > 0) {
        subOffsetByPrefix[p] = totalSize
        totalSize += 1 << subBitsByPrefix[p]
      }
    }
    this.lut = new Int32Array(totalSize).fill(-1)

    // Mark every primary slot that points to a secondary as a subtable
    // pointer up-front. Subtable leaves get filled in the second pass;
    // any primary slot that ends up with both a short-code leaf *and* a
    // pointer is an impossible case for canonical Huffman (prefix codes
    // are prefix-free), so we don't need to handle it.
    for (let p = 0; p < primarySize; p++) {
      if (subBitsByPrefix[p] > 0) {
        // Subtable pointer: sign bit = 1, bits 24..30 = subBits, bits 0..23 = offset.
        this.lut[p] = (0x80000000 | (subBitsByPrefix[p] << 24) | (subOffsetByPrefix[p] & 0xFFFFFF)) | 0
      }
    }

    // ── Second pass: fill leaves ──
    const codes = lengthsToCodes(lengths instanceof Uint8Array ? lengths : new Uint8Array(lengths))
    for (let s = 0; s < n; s++) {
      const len = (lengths as ArrayLike<number>)[s]
      if (len === 0) continue
      const code = codes[s]
      if (len <= this.lutBits) {
        // Short code: replicate across primary LUT slots that share these
        // `len` low bits.
        const reversed = reverseBits(code, len)
        const fill = 1 << (this.lutBits - len)
        const leaf = (len << 16) | s
        for (let i = 0; i < fill; i++) this.lut[reversed | (i << len)] = leaf
      } else {
        // Long code: decompose into primary prefix + secondary suffix.
        const reversed = reverseBits(code, len)
        const prefix = reversed & (primarySize - 1)
        const subOffset = subOffsetByPrefix[prefix]
        const subBits = subBitsByPrefix[prefix]
        const extraLen = len - this.lutBits
        // The secondary index is the bits *after* the primary, which on
        // the wire come *after* the primary too — same low-end bits of
        // `reversed` shifted down by `lutBits`.
        const subKey = reversed >>> this.lutBits
        const fill = 1 << (subBits - extraLen)
        // Subtable leaf encodes the *extra* length (beyond primary). The
        // reader adds `lutBits` to it before consuming.
        const leaf = (extraLen << 16) | s
        for (let i = 0; i < fill; i++) this.lut[subOffset + (subKey | (i << extraLen))] = leaf
      }
    }

    this.built = true
  }

  /**
   * Read one symbol. Two-level LUT lookup:
   *   1. peek `lutBits` → primary entry
   *   2. if leaf, consume + return
   *   3. else (subtable pointer), consume primary, peek `subBits` → leaf, consume extra
   *
   * Average case is one peek + one consume per symbol (the primary path);
   * the worst case is two peeks + two consumes for the longest codes,
   * still bounded and O(1).
   */
  readSymbol(reader: BitReader): number {
    if (!this.built) throw new Error('HuffmanTree.readSymbol called before buildFromLengths')
    const primary = reader.peek(this.lutBits)
    const entry = this.lut[primary]
    if (entry >= 0) {
      const len = (entry >>> 16) & 0xFF
      reader.consume(len)
      return entry & 0xFFFF
    }
    // Subtable lookup.
    const subBits = (entry >>> 24) & 0x7F
    const subOffset = entry & 0xFFFFFF
    reader.consume(this.lutBits)
    const subKey = reader.peek(subBits)
    const subEntry = this.lut[subOffset + subKey]
    if (subEntry < 0) {
      throw new Error('Invalid Huffman code: subtable miss')
    }
    const extraLen = (subEntry >>> 16) & 0xFF
    reader.consume(extraLen)
    return subEntry & 0xFFFF
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
