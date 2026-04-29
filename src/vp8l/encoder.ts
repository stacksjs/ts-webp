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
 *      it would help — basically always, on natural images. Encoder-side
 *      pre-transforms shrink the alphabet's entropy; the decoder un-does
 *      the transform after decoding the pixel stream.
 *   3. Tokenize: for each pixel, try (in order)
 *        a. Color cache hit  → emit a cache-index code,
 *        b. LZ77 match ≥ 3 px → emit length + distance codes,
 *        c. otherwise         → emit a literal G/R/B/A token.
 *   4. Build per-tree frequency histograms across all tokens, then build
 *      canonical Huffman code lengths for each tree.
 *   5. Write the bitstream: signature, header, transform list, color-cache
 *      flag, meta-Huffman flag, the 5 trees, then the token stream.
 *
 * Each step is gated by an option, defaulted to `true`, so we can isolate
 * regressions by encoding with one feature at a time. Round-trip is
 * verified by `decode(encode(image)) === image` for every combination.
 *
 * Reference: WebP Lossless Bitstream Specification.
 */

const VP8L_SIGNATURE = 0x2F

const NUM_LITERAL_CODES = 256
/** Total green/length symbols *before* color-cache codes are appended. */
const GREEN_BASE = NUM_LITERAL_CODES + NUM_LENGTH_CODES

/** Transform type 2 = subtract-green (no payload). */
const TRANSFORM_SUBTRACT_GREEN = 2

/** Cache bits for the color cache. 11 is the spec maximum and the size
 *  most production encoders use; the cost is one 8 KB Uint32Array. */
const DEFAULT_CACHE_BITS = 11

/** Hash multiplier per VP8L spec for the color cache. */
const VP8L_HASH_MUL = 0x1E35A7BD

/**
 * Internal options for fine-grained control over which features to apply.
 * Public callers go through `WebpEncodeOptions` (defined in `../types`),
 * which currently only exposes `lossless` / `quality` / `effort` / `alpha`;
 * the per-feature toggles below are used by tests to validate each
 * transform in isolation.
 */
interface InternalOptions {
  subtractGreen: boolean
  useLZ77: boolean
  useColorCache: boolean
  cacheBits: number
}

function resolveOptions(opts: WebpEncodeOptions): InternalOptions {
  // Honour both the documented options and any internal flags a caller
  // (e.g. our test suite) might smuggle through.
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
  const tokens = tokenize(argb, width, opts)

  // ── Step 4: histograms + trees ──
  const cacheSize = opts.useColorCache ? 1 << opts.cacheBits : 0
  const greenAlphabet = GREEN_BASE + cacheSize

  const greenFreq = new Uint32Array(greenAlphabet)
  const redFreq = new Uint32Array(256)
  const blueFreq = new Uint32Array(256)
  const alphaFreq = new Uint32Array(256)
  const distFreq = new Uint32Array(NUM_DISTANCE_CODES)

  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i]
    switch (t.kind) {
      case TokenKind.Literal:
        greenFreq[t.g]++
        redFreq[t.r]++
        blueFreq[t.b]++
        alphaFreq[t.a]++
        break
      case TokenKind.Backref:
        greenFreq[NUM_LITERAL_CODES + t.lengthCode]++
        distFreq[t.distanceCode]++
        break
      case TokenKind.Cache:
        greenFreq[GREEN_BASE + t.index]++
        break
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

  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i]
    switch (t.kind) {
      case TokenKind.Literal: {
        emitSymbol(writer, greenLen[t.g], greenCodes[t.g])
        emitSymbol(writer, redLen[t.r], redCodes[t.r])
        emitSymbol(writer, blueLen[t.b], blueCodes[t.b])
        emitSymbol(writer, alphaLen[t.a], alphaCodes[t.a])
        break
      }
      case TokenKind.Backref: {
        const greenSym = NUM_LITERAL_CODES + t.lengthCode
        emitSymbol(writer, greenLen[greenSym], greenCodes[greenSym])
        // Length extra bits — already validated to fit in 24 bits.
        if (t.lengthExtraBits > 0) writer.writeBits(t.lengthExtraValue, t.lengthExtraBits)
        emitSymbol(writer, distLen[t.distanceCode], distCodes[t.distanceCode])
        if (t.distanceExtraBits > 0) writer.writeBits(t.distanceExtraValue, t.distanceExtraBits)
        break
      }
      case TokenKind.Cache: {
        const greenSym = GREEN_BASE + t.index
        emitSymbol(writer, greenLen[greenSym], greenCodes[greenSym])
        break
      }
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

const enum TokenKind {
  Literal = 0,
  Backref = 1,
  Cache = 2,
}

interface LiteralToken { kind: TokenKind.Literal, g: number, r: number, b: number, a: number }
interface BackrefToken {
  kind: TokenKind.Backref
  lengthCode: number
  lengthExtraBits: number
  lengthExtraValue: number
  distanceCode: number
  distanceExtraBits: number
  distanceExtraValue: number
}
interface CacheToken { kind: TokenKind.Cache, index: number }
type Token = LiteralToken | BackrefToken | CacheToken

/**
 * Tokenize the ARGB pixel stream into a Token[] suitable for histograms +
 * emission. Honours the encoder option flags to enable/disable LZ77 and
 * the color cache independently — useful for debugging and for tests
 * that want to verify individual compression stages in isolation.
 */
function tokenize(argb: Uint32Array, width: number, opts: InternalOptions): Token[] {
  const tokens: Token[] = []
  const n = argb.length

  // Color cache: maps a hashed slot to the most recent ARGB value put
  // there. A "hit" is when the slot's stored value equals the pixel we
  // were about to emit — we then skip the literal emission and instead
  // emit a cache-index code, which is much shorter.
  const cacheBits = opts.cacheBits
  const cacheSize = opts.useColorCache ? 1 << cacheBits : 0
  const cache = cacheSize > 0 ? new Uint32Array(cacheSize) : null
  // Track whether each slot has been populated yet (zero is a valid pixel
  // so we can't use a sentinel value).
  const cacheValid = cacheSize > 0 ? new Uint8Array(cacheSize) : null

  // LZ77: 16-bit hash over 3-pixel ARGB windows. Single-position-per-hash
  // chaining (the equivalent of DEFLATE's "level 1") — fast enough that
  // most natural images encode in a few ms.
  const HASH_BITS = 16
  const HASH_SIZE = 1 << HASH_BITS
  const HASH_MASK = HASH_SIZE - 1
  const hashTable = opts.useLZ77 ? new Int32Array(HASH_SIZE).fill(-1) : null

  let i = 0
  while (i < n) {
    const px = argb[i]

    // Try LZ77 first — a long-run match beats N cache hits, even though
    // cache hits are cheap individually. Only after LZ77 declines do we
    // fall to the cache → literal cascade.
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
        ) {
          len++
        }
        if (len >= 3) {
          const distance = i - matchPos
          const distEnc = distanceToCode(distance)
          const lenEnc = lengthToCode(len)
          if (distEnc !== null && lenEnc !== null) {
            tokens.push({
              kind: TokenKind.Backref,
              lengthCode: lenEnc.code,
              lengthExtraBits: lenEnc.extraBits,
              lengthExtraValue: lenEnc.extraValue,
              distanceCode: distEnc.code,
              distanceExtraBits: distEnc.extraBits,
              distanceExtraValue: distEnc.extraValue,
            })
            // Update cache + hash for every pixel in the match so the
            // decoder's view stays in sync and later matches can pick
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

    // No LZ77 match — try color-cache hit. This is a single-pixel
    // operation, so it doesn't compete with multi-pixel matches; it
    // only wins where LZ77 didn't.
    if (cache && cacheValid) {
      const slot = (Math.imul(px, VP8L_HASH_MUL) >>> (32 - cacheBits)) & (cacheSize - 1)
      if (cacheValid[slot] && cache[slot] === px) {
        tokens.push({ kind: TokenKind.Cache, index: slot })
        i++
        continue
      }
    }

    // Fall back to literal emission.
    const a = (px >>> 24) & 0xFF
    const r = (px >>> 16) & 0xFF
    const g = (px >>> 8) & 0xFF
    const b = px & 0xFF
    tokens.push({ kind: TokenKind.Literal, g, r, b, a })
    if (cache && cacheValid) {
      const slot = (Math.imul(px, VP8L_HASH_MUL) >>> (32 - cacheBits)) & (cacheSize - 1)
      cache[slot] = px
      cacheValid[slot] = 1
    }
    i++
  }

  return tokens
}

/**
 * VP8L's 3-pixel hash. Mixes the three ARGB words with multiplicative
 * hashing; we only need a uniform distribution over `hashMask`, not
 * cryptographic strength.
 */
function hashOf(argb: Uint32Array, i: number, hashMask: number): number {
  // Combine three Uint32s with imul-based mixing.
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
