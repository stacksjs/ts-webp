import type { WebpEncodeOptions, WebpImageData } from '../types'
import { BitWriter } from '../bitreader'
import { distanceToCode, linearDistanceToRaw, NUM_DISTANCE_CODES } from './distance'
import { buildCodeLengths, lengthsToCodes, reverseBits, writeHuffmanTree } from './huffman'
import { lengthToCode, MAX_LENGTH, NUM_LENGTH_CODES } from './length'
import { applyColorTransform } from './color'
import { applyColorIndexTransform, deltaEncodePalette, MAX_PALETTE_SIZE } from './color-index'
import { applyPredictorTransform } from './predictor'

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

/** Transform type 0 = predictor (3-bit size_bits + recursive sub-image). */
const TRANSFORM_PREDICTOR = 0
/** Transform type 1 = color (3-bit size_bits + per-block coefficient sub-image). */
const TRANSFORM_COLOR = 1
/** Transform type 2 = subtract-green (no payload). */
const TRANSFORM_SUBTRACT_GREEN = 2
/** Transform type 3 = color-indexing (palette + packed indices). */
const TRANSFORM_COLOR_INDEXING = 3

/** Cache bits for the color cache. 11 is the spec maximum. */
const DEFAULT_CACHE_BITS = 11

/** Hash multiplier per VP8L spec for the color cache. */
const VP8L_HASH_MUL = 0x1E35A7BD

/** Token discriminator: literal pixel, LZ77 backreference, color-cache hit. */
enum TokenKind {
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
  predictor: boolean
  /** Block size as `1 << predictorBits`. Spec range: 2..7 (block 4..128). */
  predictorBits: number
  /** Per-block color transform. Smaller win than predictor; off by default
   *  because the encoder's coefficient search is moderately expensive. */
  color: boolean
  colorBits: number
  /**
   * When true, the encoder probes for ≤ 256 distinct colours and (if so)
   * uses the color-indexing transform instead of SG/predictor. Enabled by
   * default — it's free to probe and a massive win on palette images.
   */
  useColorIndex: boolean
  useLZ77: boolean
  useColorCache: boolean
  cacheBits: number
}

function resolveOptions(opts: WebpEncodeOptions): InternalOptions {
  const o = opts as WebpEncodeOptions & Partial<InternalOptions>
  return {
    subtractGreen: o.subtractGreen ?? true,
    // Predictor: enabled by default. The encoder's all-blocks-mode-11
    // strategy gets us most of the available win at one extra pixel pass
    // and a handful of extra bits in the bitstream.
    predictor: o.predictor ?? true,
    predictorBits: o.predictorBits ?? 4,
    // Color transform: off by default. It typically adds 2-5 % over
    // SG+predictor on natural images, but the encoder-side coefficient
    // search is moderately expensive (5×5×5 candidates per block) and
    // the win can be eaten by the per-block coefficient encoding
    // overhead on small images.
    color: o.color ?? false,
    colorBits: o.colorBits ?? 4,
    useColorIndex: o.useColorIndex ?? true,
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

  // ── Step 2: pre-transforms ──
  // We pick one of two paths based on the input:
  //   • If the image has ≤ 256 distinct colours: color-indexing alone.
  //     A palette image needs no per-pixel prediction or channel
  //     decorrelation, and color-indexing's packed-indices form is
  //     dramatically smaller than even the best Huffman over the
  //     original pixel stream.
  //   • Otherwise: SG + predictor stack (the photo path).
  // We don't combine the two — color-indexed pixels have R=B=0 and
  // alpha=0xFF, which makes SG a no-op and predictor's win marginal,
  // while still adding bitstream metadata overhead. Real-world wins
  // come from picking the *right* path per image, not stacking them.
  let workArgb: Uint32Array = argb
  let workWidth = width
  let usedSubtractGreen = false
  let predictorInfo: { modeImage: Uint32Array, modeWidth: number, modeHeight: number } | null = null
  let colorIndexInfo: { palette: Uint32Array, encodedWidth: number } | null = null

  if (opts.useColorIndex) {
    const result = applyColorIndexTransform(argb, width, height)
    // Only commit to color-indexing when the palette is small enough
    // that its packing actually beats the SG/predictor stack. The
    // packing-factor table caps useful packing at palette ≤ 16; above
    // that, color-indexing's only saving is a smaller green alphabet,
    // which loses to the predictor on natural-image content. For smooth
    // photo-like inputs that happen to have ≤ 256 distinct colours
    // (e.g. small images with low-amplitude sin/cos blends), the
    // SG+predictor path wins by a wide margin — so we only take CI
    // when its packing is non-trivial.
    if (result !== null && result.palette.length <= 16) {
      workArgb = result.encodedArgb
      workWidth = result.encodedWidth
      colorIndexInfo = { palette: result.palette, encodedWidth: result.encodedWidth }
    }
  }

  let colorInfo: { ctImage: Uint32Array, ctWidth: number, ctHeight: number } | null = null
  if (!colorIndexInfo) {
    // Application order: color first, then SG, then predictor. SG depends
    // on green being roughly representative of the block; if color has
    // already shifted red/blue, SG operates on the residuals which keeps
    // the channels small.
    if (opts.color && workWidth >= 2 && height >= 2) {
      colorInfo = applyColorTransform(workArgb, workWidth, height, opts.colorBits)
    }
    if (opts.subtractGreen) {
      applySubtractGreen(workArgb)
      usedSubtractGreen = true
    }
    if (opts.predictor && workWidth >= 2 && height >= 2) {
      predictorInfo = applyPredictorTransform(workArgb, workWidth, height, opts.predictorBits)
    }
  }

  // ── Step 3: tokenize ──
  // Tokenize the post-transform image. Note that workArgb may be smaller
  // than the original (color-indexing reduces width), so the actual
  // tokenization size is `workArgb.length`, not `numPixels`.
  const workNumPixels = workArgb.length
  const tokenKind = new Uint8Array(workNumPixels)
  const tokenA = new Uint32Array(workNumPixels)
  const tokenB = new Uint32Array(workNumPixels)
  const numTokens = tokenize(workArgb, workWidth, opts, tokenKind, tokenA, tokenB)

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

  // Spec: transforms appear in the bitstream in the *forward* order they
  // were applied. The decoder reads them in order then applies inverses
  // in reverse, which unwinds the encoder's stack.
  //
  // We use one of two transform stacks at most:
  //   • [color-indexing]              for palette images
  //   • [subtract-green, predictor]   for photo content
  // Mixing the two doesn't help (see Step 2), so the bitstream emits at
  // most one stack.
  if (colorIndexInfo) {
    writer.writeBit(1) // transform-present
    writer.writeBits(TRANSFORM_COLOR_INDEXING, 2)
    writer.writeBits(colorIndexInfo.palette.length - 1, 8)
    // Palette is delta-encoded on the wire. The decoder undoes the
    // delta after reading the palette sub-image.
    const deltaPalette = deltaEncodePalette(colorIndexInfo.palette)
    writeSubImage(writer, deltaPalette, deltaPalette.length, 1)
  } else {
    // Photo-path order: color, then SG, then predictor — same order as
    // applied. The decoder reads them in this order and applies inverse
    // in reverse (predictor, then SG, then color), which unwinds the
    // encoder stack correctly.
    if (colorInfo) {
      writer.writeBit(1)
      writer.writeBits(TRANSFORM_COLOR, 2)
      writer.writeBits(opts.colorBits - 2, 3)
      writeSubImage(writer, colorInfo.ctImage, colorInfo.ctWidth, colorInfo.ctHeight)
    }
    if (usedSubtractGreen) {
      writer.writeBit(1) // transform-present
      writer.writeBits(TRANSFORM_SUBTRACT_GREEN, 2)
    }
    if (predictorInfo) {
      writer.writeBit(1) // transform-present
      writer.writeBits(TRANSFORM_PREDICTOR, 2)
      writer.writeBits(opts.predictorBits - 2, 3)
      writeSubImage(writer, predictorInfo.modeImage, predictorInfo.modeWidth, predictorInfo.modeHeight)
    }
  }
  writer.writeBit(0) // end of transform chain
  void MAX_PALETTE_SIZE // reference so import isn't dead

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

  // After the simple-code header is on the wire, zero-out lengths for
  // single-symbol trees so per-pixel `emitSymbol` calls contribute no
  // bits. The decoder side (`HuffmanTree` for a 1-leaf tree) consumes
  // 0 bits per read, so encoder and decoder stay in sync.
  zeroIfSingleSymbol(greenLen)
  zeroIfSingleSymbol(redLen)
  zeroIfSingleSymbol(blueLen)
  zeroIfSingleSymbol(alphaLen)
  zeroIfSingleSymbol(distLen)

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
  width: number,
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
          // Convert the linear distance into the cheapest raw value (plane
          // code or `distance + 120`) before prefix-encoding it.
          const rawDistance = linearDistanceToRaw(distance, width)
          const distEnc = rawDistance !== null ? distanceToCode(rawDistance) : null
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
  // `len === 0` is the "single-symbol tree" signal: the simple-code header
  // already pinned which symbol every read returns, so per-pixel emits
  // contribute zero bits to the stream. Mirrors libwebp's decode-side
  // behaviour (`HuffmanCode.bits = 0` after `BuildHuffmanTable` for a
  // 1-leaf tree).
  if (len === 0) return
  writer.writeBits(reverseBits(code, len), len)
}

function sum(arr: Uint32Array): number {
  let s = 0
  for (let i = 0; i < arr.length; i++) s += arr[i]
  return s
}

/**
 * If `lengths` contains exactly one non-zero entry (a single-symbol tree),
 * zero it. The simple-code wire header already pins the symbol; per-pixel
 * `emitSymbol` calls should contribute zero bits, matching libwebp's
 * decode-side behaviour.
 */
function zeroIfSingleSymbol(lengths: Uint8Array): void {
  let nonZero = 0
  let firstIdx = -1
  for (let i = 0; i < lengths.length; i++) {
    if (lengths[i] > 0) {
      nonZero++
      if (firstIdx < 0) firstIdx = i
      if (nonZero > 1) return
    }
  }
  if (nonZero === 1) lengths[firstIdx] = 0
}

// ---------------------------------------------------------------------------
// Sub-image emit (for predictor mode images and color-transform images)
// ---------------------------------------------------------------------------

/**
 * Emit a small ARGB image as a VP8L "entropy image" — the same format
 * VP8L uses for the main image, but without the outer signature/header,
 * without nested transforms, and (in our implementation) without LZ77
 * or color cache. Just five Huffman trees over the per-channel literals
 * plus literal-only pixel data.
 *
 * Used for the mode-per-block image of the predictor transform and the
 * color-transform image. Both are tiny (typically a few hundred pixels)
 * with a small set of distinct values, so literal-only encoding plus
 * Huffman is already near-optimal — adding LZ77 here would cost more
 * bitstream overhead than it'd save.
 */
function writeSubImage(writer: BitWriter, argb: Uint32Array, width: number, height: number): void {
  // Sub-image header — mirrors libwebp's `DecodeImageStream(...,
  // is_level0=0, ...)` exactly. Sub-images skip the top-level
  // transform-present and meta-Huffman bits (those are gated by
  // `is_level0`); they go straight to the (here always absent)
  // color-cache flag.
  writer.writeBit(0) // color-cache present = 0

  const numPixels = width * height
  const greenFreq = new Uint32Array(GREEN_BASE)
  const redFreq = new Uint32Array(256)
  const blueFreq = new Uint32Array(256)
  const alphaFreq = new Uint32Array(256)
  // Distance tree must still be present even though we emit no backrefs.
  // Pin one symbol so it gets a well-formed (length-1) tree.
  const distFreq = new Uint32Array(NUM_DISTANCE_CODES)
  distFreq[0] = 1

  for (let i = 0; i < numPixels; i++) {
    const px = argb[i]
    alphaFreq[(px >>> 24) & 0xFF]++
    redFreq[(px >>> 16) & 0xFF]++
    greenFreq[(px >>> 8) & 0xFF]++
    blueFreq[px & 0xFF]++
  }

  // Each tree must have ≥ 1 non-zero symbol (else simple-code path can't
  // fire). Empty channels get a placeholder so tree construction works.
  if (sum(redFreq) === 0) redFreq[0] = 1
  if (sum(blueFreq) === 0) blueFreq[0] = 1
  if (sum(alphaFreq) === 0) alphaFreq[0] = 1
  if (sum(greenFreq) === 0) greenFreq[0] = 1

  const greenLen = buildCodeLengths(greenFreq)
  const redLen = buildCodeLengths(redFreq)
  const blueLen = buildCodeLengths(blueFreq)
  const alphaLen = buildCodeLengths(alphaFreq)
  const distLen = buildCodeLengths(distFreq)

  writeHuffmanTree(writer, greenLen)
  writeHuffmanTree(writer, redLen)
  writeHuffmanTree(writer, blueLen)
  writeHuffmanTree(writer, alphaLen)
  writeHuffmanTree(writer, distLen)

  // Match libwebp's 0-bits-per-read decode for single-symbol trees.
  zeroIfSingleSymbol(greenLen)
  zeroIfSingleSymbol(redLen)
  zeroIfSingleSymbol(blueLen)
  zeroIfSingleSymbol(alphaLen)
  zeroIfSingleSymbol(distLen)

  const greenCodes = lengthsToCodes(greenLen)
  const redCodes = lengthsToCodes(redLen)
  const blueCodes = lengthsToCodes(blueLen)
  const alphaCodes = lengthsToCodes(alphaLen)

  for (let i = 0; i < numPixels; i++) {
    const px = argb[i]
    const a = (px >>> 24) & 0xFF
    const r = (px >>> 16) & 0xFF
    const g = (px >>> 8) & 0xFF
    const b = px & 0xFF
    emitSymbol(writer, greenLen[g], greenCodes[g])
    emitSymbol(writer, redLen[r], redCodes[r])
    emitSymbol(writer, blueLen[b], blueCodes[b])
    emitSymbol(writer, alphaLen[a], alphaCodes[a])
  }
}
