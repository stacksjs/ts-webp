import type { WebpEncodeOptions, WebpImageData } from '../types'
import { BitWriter } from '../bitreader'
import { distanceToCode, NUM_DISTANCE_CODES } from './distance'
import { buildCodeLengths, lengthsToCodes, reverseBits, writeHuffmanTree } from './huffman'
import { lengthToCode, MAX_LENGTH, NUM_LENGTH_CODES } from './length'

/**
 * VP8L (lossless WebP) encoder.
 *
 * Pipeline:
 *
 *   1. Pack RGBA bytes into ARGB Uint32Array (one word per pixel).
 *   2. Apply `subtract-green` pre-transform (R -= G, B -= G mod 256) when
 *      enabled. The decoder un-does it after pixel decode.
 *   3. Tokenize: for each pixel, try (in order)
 *        a. LZ77 match ≥ 3 px → emit length + distance codes,
 *        b. Color cache hit  → emit a cache-index code,
 *        c. otherwise         → emit a literal G/R/B/A token.
 *   4. Build per-tree frequency histograms across all tokens, then build
 *      canonical Huffman code lengths for each tree.
 *   5. Write the bitstream: signature, header, transform list, color-cache
 *      flag, meta-Huffman flag, the 5 trees, then the token stream.
 *
 * Tokens are kept in three parallel typed arrays (`tokenKind`, `tokenA`,
 * `tokenB`) sized at `numPixels` — that's the maximum possible token
 * count. The previous implementation used a `Token` discriminated-union
 * `Array`, which allocated one object per pixel and produced ~50-70 MB
 * of GC pressure on a 1 MP image. The flat-array encoding keeps the
 * working set under 9 MB and lets the histogram + emit passes iterate
 * with simple field reads.
 *
 * Reference: WebP Lossless Bitstream Specification.
 */

const VP8L_SIGNATURE = 0x2F

const NUM_LITERAL_CODES = 256
/** Total green/length symbols *before* color-cache codes are appended. */
const GREEN_BASE = NUM_LITERAL_CODES + NUM_LENGTH_CODES

/** Transform type 2 = subtract-green (no payload). */
const TRANSFORM_SUBTRACT_GREEN = 2

/** Cache bits for the color cache. 11 is the spec maximum. */
const DEFAULT_CACHE_BITS = 11

/** Hash multiplier per VP8L spec for the color cache. */
const VP8L_HASH_MUL = 0x1E35A7BD

/** Token discriminator: literal pixel, LZ77 backreference, color-cache hit. */
const enum TokenKind {
  Literal = 0,
  Backref = 1,
  Cache = 2,
}

/**
 * Internal options for fine-grained control over which features to apply.
 * Public callers go through `WebpEncodeOptions`; the per-feature toggles
 * here are used by tests to validate each transform in isolation.
 */
interface InternalOptions {
  subtractGreen: boolean
  useLZ77: boolean
  useColorCache: boolean
  cacheBits: number
}

function resolveOptions(opts: WebpEncodeOptions): InternalOptions {
  const o = opts as WebpEncodeOptions & Partial<InternalOptions>
  return {
    subtractGreen: o.subtractGreen ?? true,
    useLZ77: o.useLZ77 ?? true,
    useColorCache: o.useColorCache ?? true,
    cacheBits: o.cacheBits ?? DEFAULT_CACHE_BITS,
  }
}

export function encodeVP8L(
  imageData: WebpImageData,
  options: WebpEncodeOptions = {},
): Uint8Array {
  const { data, width, height } = imageData
  if (width <= 0 || height <= 0) throw new Error('VP8L: width and height must be positive')
  if (width > 16384 || height > 16384) throw new Error('VP8L: dimensions cap at 16384')
  if (data.length < width * height * 4) throw new Error('VP8L: pixel buffer is shorter than width × height × 4')

  const opts = resolveOptions(options)
  const numPixels = width * height

  // ── Step 1: pack RGBA → ARGB words ──
  const argb = new Uint32Array(numPixels)
  for (let i = 0; i < numPixels; i++) {
    const o = i * 4
    argb[i] = (data[o + 3] << 24) | (data[o] << 16) | (data[o + 1] << 8) | data[o + 2]
  }

  // ── Step 2: subtract-green ──
  if (opts.subtractGreen) applySubtractGreen(argb)

  // ── Step 3: tokenize ──
  // Worst case = one token per pixel (all literals). We allocate parallel
  // typed arrays at that size and track the actual count separately.
  // This avoids ~50-70 MB of GC pressure on 1 MP+ images.
  const tokenKind = new Uint8Array(numPixels)
  const tokenA = new Uint32Array(numPixels)
  const tokenB = new Uint32Array(numPixels)
  const numTokens = tokenize(argb, opts, tokenKind, tokenA, tokenB)

  // ── Step 4: histograms + trees ──
  const cacheSize = opts.useColorCache ? 1 << opts.cacheBits : 0
  const greenAlphabet = GREEN_BASE + cacheSize

  const greenFreq = new Uint32Array(greenAlphabet)
  const redFreq = new Uint32Array(256)
  const blueFreq = new Uint32Array(256)
  const alphaFreq = new Uint32Array(256)
  const distFreq = new Uint32Array(NUM_DISTANCE_CODES)

  for (let i = 0; i < numTokens; i++) {
    const kind = tokenKind[i]
    const a = tokenA[i]
    if (kind === TokenKind.Literal) {
      // Literal token: tokenA holds (g << 24) | (r << 16) | (b << 8) | alpha.
      // Choosing this packing keeps `g` (the alphabet's most-touched
      // symbol) in the high byte where the JIT can't accidentally
      // sign-extend.
      greenFreq[(a >>> 24) & 0xFF]++
      redFreq[(a >>> 16) & 0xFF]++
      blueFreq[(a >>> 8) & 0xFF]++
      alphaFreq[a & 0xFF]++
    } else if (kind === TokenKind.Backref) {
      // Backref: tokenA = lengthCode | (lengthExtraBits << 8) | (lengthExtraValue << 16)
      //          tokenB = distanceCode | (distanceExtraBits << 8) | (distanceExtraValue << 16)
      greenFreq[NUM_LITERAL_CODES + (a & 0xFF)]++
      distFreq[tokenB[i] & 0xFF]++
    } else {
      // Cache: tokenA = cache index.
      greenFreq[GREEN_BASE + a]++
    }
  }

  // VP8L requires every tree to have at least one valid code. If a tree
  // has zero entries (e.g. distance tree when LZ77 emits no backrefs),
  // pin a placeholder symbol so `buildCodeLengths` returns a length-1
  // tree we can encode via the simple-code path.
  if (sum(distFreq) === 0) distFreq[0] = 1
  if (sum(redFreq) === 0) redFreq[0] = 1
  if (sum(blueFreq) === 0) blueFreq[0] = 1
  if (sum(alphaFreq) === 0) alphaFreq[0] = 1
  if (sum(greenFreq) === 0) greenFreq[0] = 1

  const greenLen = buildCodeLengths(greenFreq)
  const redLen = buildCodeLengths(redFreq)
  const blueLen = buildCodeLengths(blueFreq)
  const alphaLen = buildCodeLengths(alphaFreq)
  const distLen = buildCodeLengths(distFreq)

  // ── Step 5: emit bitstream ──
  const writer = new BitWriter(numPixels * 4 + 1024)

  writer.writeBits(VP8L_SIGNATURE, 8)
  writer.writeBits(width - 1, 14)
  writer.writeBits(height - 1, 14)
  writer.writeBit(imageData.hasAlpha ? 1 : 0)
  writer.writeBits(0, 3) // version

  if (opts.subtractGreen) {
    writer.writeBit(1) // transform-present
    writer.writeBits(TRANSFORM_SUBTRACT_GREEN, 2)
  }
  writer.writeBit(0) // end of transform chain

  if (opts.useColorCache) {
    writer.writeBit(1) // color-cache present
    writer.writeBits(opts.cacheBits, 4)
  } else {
    writer.writeBit(0)
  }

  writer.writeBit(0) // no meta-Huffman image

  writeHuffmanTree(writer, greenLen)
  writeHuffmanTree(writer, redLen)
  writeHuffmanTree(writer, blueLen)
  writeHuffmanTree(writer, alphaLen)
  writeHuffmanTree(writer, distLen)

  // Pre-compute canonical codes for fast lookup during emission.
  const greenCodes = lengthsToCodes(greenLen)
  const redCodes = lengthsToCodes(redLen)
  const blueCodes = lengthsToCodes(blueLen)
  const alphaCodes = lengthsToCodes(alphaLen)
  const distCodes = lengthsToCodes(distLen)

  for (let i = 0; i < numTokens; i++) {
    const kind = tokenKind[i]
    const a = tokenA[i]
    if (kind === TokenKind.Literal) {
      const g = (a >>> 24) & 0xFF
      const r = (a >>> 16) & 0xFF
      const b = (a >>> 8) & 0xFF
      const al = a & 0xFF
      emitSymbol(writer, greenLen[g], greenCodes[g])
      emitSymbol(writer, redLen[r], redCodes[r])
      emitSymbol(writer, blueLen[b], blueCodes[b])
      emitSymbol(writer, alphaLen[al], alphaCodes[al])
    } else if (kind === TokenKind.Backref) {
      const lengthCode = a & 0xFF
      const lengthExtraBits = (a >>> 8) & 0xFF
      const lengthExtraValue = (a >>> 16) & 0xFFFF
      const bb = tokenB[i]
      const distanceCode = bb & 0xFF
      const distanceExtraBits = (bb >>> 8) & 0xFF
      const distanceExtraValue = (bb >>> 16) & 0xFFFF
      const greenSym = NUM_LITERAL_CODES + lengthCode
      emitSymbol(writer, greenLen[greenSym], greenCodes[greenSym])
      if (lengthExtraBits > 0) writer.writeBits(lengthExtraValue, lengthExtraBits)
      emitSymbol(writer, distLen[distanceCode], distCodes[distanceCode])
      if (distanceExtraBits > 0) writer.writeBits(distanceExtraValue, distanceExtraBits)
    } else {
      const greenSym = GREEN_BASE + a
      emitSymbol(writer, greenLen[greenSym], greenCodes[greenSym])
    }
  }

  return writer.getBuffer()
}

// ---------------------------------------------------------------------------
// Subtract-green
// ---------------------------------------------------------------------------

function applySubtractGreen(argb: Uint32Array): void {
  for (let i = 0; i < argb.length; i++) {
    const px = argb[i]
    const a = px & 0xFF000000
    const r = (px >>> 16) & 0xFF
    const g = (px >>> 8) & 0xFF
    const b = px & 0xFF
    const r2 = (r - g) & 0xFF
    const b2 = (b - g) & 0xFF
    argb[i] = (a | (r2 << 16) | (g << 8) | b2) >>> 0
  }
}

// ---------------------------------------------------------------------------
// Tokenization (LZ77 + color cache + literal)
// ---------------------------------------------------------------------------

/**
 * Single forward pass over the ARGB buffer, emitting one token per
 * "consumed pixel block". Tokens are written to caller-provided typed
 * arrays — this lets callers reuse buffers across encodes.
 *
 * Returns the number of tokens emitted.
 *
 * Token packing:
 *   `tokenKind[i]` = 0 (literal) | 1 (backref) | 2 (cache)
 *   Literal: tokenA = (g << 24) | (r << 16) | (b << 8) | a
 *   Backref: tokenA = lengthCode | (lengthExtraBits << 8) | (lengthExtraValue << 16)
 *            tokenB = distanceCode | (distanceExtraBits << 8) | (distanceExtraValue << 16)
 *   Cache:   tokenA = cache index
 */
function tokenize(
  argb: Uint32Array,
  opts: InternalOptions,
  tokenKind: Uint8Array,
  tokenA: Uint32Array,
  tokenB: Uint32Array,
): number {
  const n = argb.length
  let nTokens = 0

  // Color cache.
  const cacheBits = opts.cacheBits
  const cacheSize = opts.useColorCache ? 1 << cacheBits : 0
  const cache = cacheSize > 0 ? new Uint32Array(cacheSize) : null
  const cacheValid = cacheSize > 0 ? new Uint8Array(cacheSize) : null

  // LZ77 hash table.
  const HASH_BITS = 16
  const HASH_SIZE = 1 << HASH_BITS
  const HASH_MASK = HASH_SIZE - 1
  const hashTable = opts.useLZ77 ? new Int32Array(HASH_SIZE).fill(-1) : null

  let i = 0
  while (i < n) {
    const px = argb[i]

    // ── LZ77 first ──
    if (hashTable && i + 2 < n) {
      const h = hashOf(argb, i, HASH_MASK)
      const matchPos = hashTable[h]
      hashTable[h] = i

      if (matchPos >= 0 && i - matchPos > 0 && i - matchPos < 1 << 20) {
        let len = 0
        while (
          len < MAX_LENGTH
          && i + len < n
          && argb[matchPos + len] === argb[i + len]
        ) len++
        if (len >= 3) {
          const distance = i - matchPos
          const distEnc = distanceToCode(distance)
          const lenEnc = lengthToCode(len)
          if (distEnc !== null && lenEnc !== null) {
            tokenKind[nTokens] = TokenKind.Backref
            tokenA[nTokens] = lenEnc.code | (lenEnc.extraBits << 8) | (lenEnc.extraValue << 16)
            tokenB[nTokens] = distEnc.code | (distEnc.extraBits << 8) | (distEnc.extraValue << 16)
            nTokens++
            // Update cache + hash for every pixel inside the run so the
            // decoder's cache stays in sync and later matches can pick
            // up from inside this run.
            if (cache && cacheValid) {
              for (let k = 0; k < len && i + k < n; k++) {
                const cpx = argb[i + k]
                const slot = (Math.imul(cpx, VP8L_HASH_MUL) >>> (32 - cacheBits)) & (cacheSize - 1)
                cache[slot] = cpx
                cacheValid[slot] = 1
              }
            }
            for (let k = 1; k < len && i + k + 2 < n; k++) {
              hashTable[hashOf(argb, i + k, HASH_MASK)] = i + k
            }
            i += len
            continue
          }
        }
      }
    }

    // ── Color cache hit ──
    if (cache && cacheValid) {
      const slot = (Math.imul(px, VP8L_HASH_MUL) >>> (32 - cacheBits)) & (cacheSize - 1)
      if (cacheValid[slot] && cache[slot] === px) {
        tokenKind[nTokens] = TokenKind.Cache
        tokenA[nTokens] = slot
        nTokens++
        i++
        continue
      }
    }

    // ── Literal ──
    const a = (px >>> 24) & 0xFF
    const r = (px >>> 16) & 0xFF
    const g = (px >>> 8) & 0xFF
    const b = px & 0xFF
    tokenKind[nTokens] = TokenKind.Literal
    tokenA[nTokens] = (g << 24) | (r << 16) | (b << 8) | a
    nTokens++
    if (cache && cacheValid) {
      const slot = (Math.imul(px, VP8L_HASH_MUL) >>> (32 - cacheBits)) & (cacheSize - 1)
      cache[slot] = px
      cacheValid[slot] = 1
    }
    i++
  }

  return nTokens
}

/**
 * VP8L's 3-pixel hash. Mixes three ARGB words via multiplicative hashing;
 * we only need uniform distribution over `hashMask`, not crypto strength.
 */
function hashOf(argb: Uint32Array, i: number, hashMask: number): number {
  const a = argb[i]
  const b = argb[i + 1]
  const c = argb[i + 2]
  let h = Math.imul(a, 0x9E3779B1)
  h = Math.imul(h ^ b, 0x9E3779B1)
  h = Math.imul(h ^ c, 0x9E3779B1)
  return (h >>> 0) & hashMask
}

// ---------------------------------------------------------------------------
// Bitstream emit helpers
// ---------------------------------------------------------------------------

function emitSymbol(writer: BitWriter, len: number, code: number): void {
  if (len === 0) {
    throw new Error('emitSymbol: zero-length code for an emitted symbol')
  }
  writer.writeBits(reverseBits(code, len), len)
}

function sum(arr: Uint32Array): number {
  let s = 0
  for (let i = 0; i < arr.length; i++) s += arr[i]
  return s
}
