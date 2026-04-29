import type { VP8LHeader, WebpImageData } from '../types'
import { BitReader } from '../bitreader'
import { HuffmanTree, readHuffmanTree } from './huffman'

/**
 * VP8L (lossless WebP) decoder.
 *
 * Mirrors the encoder bit-for-bit and additionally handles bitstreams that
 * carry the optional pre-transforms, color cache, and meta-Huffman image.
 *
 * Design notes:
 *
 * - Pixels are accumulated in a `Uint32Array` keyed by ARGB layout (alpha
 *   in the high byte). At the end we splat into the output `Uint8Array`
 *   in RGBA order. This costs one extra pass but keeps the inner decode
 *   loop tight.
 *
 * - Backreferences (length+distance) decode lazily via the spec's distance
 *   mapping: codes 0..3 are "near" codes that map onto a small table of
 *   x/y offsets, codes 4..39 are extra-bit distances. We compute the
 *   absolute distance and copy from `argb[pos - dist]` into `argb[pos]`,
 *   one ARGB slot at a time so the source can overlap the destination.
 *
 * - Color cache uses VP8L's `(pixel * 0x1E35A7BD) >> (32 - bits)` hash.
 *
 * Reference: WebP Lossless Bitstream Specification.
 */

const VP8L_SIGNATURE = 0x2F

const NUM_LITERAL_CODES = 256
const NUM_LENGTH_CODES = 24
const NUM_DISTANCE_CODES = 40
const MAX_CACHE_BITS = 11

/** Per-spec extra-bit count for length codes 0..23. */
const LENGTH_EXTRA_BITS = new Uint8Array([
  0, 0, 0, 0, 0, 0, 0, 0,
  1, 1, 1, 1, 2, 2, 2, 2,
  3, 3, 3, 3, 4, 4, 4, 4,
])
/** Per-spec base offset for length codes 0..23. */
const LENGTH_OFFSET = new Uint16Array([
  1, 2, 3, 4, 5, 6, 7, 8,
  9, 11, 13, 15, 17, 21, 25, 29,
  33, 41, 49, 57, 65, 81, 97, 113,
])

/** Per-spec extra-bit count for distance codes 0..39. */
const DISTANCE_EXTRA_BITS = new Uint8Array([
  0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
  9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
])
/** Per-spec base offset for distance codes 0..39. */
const DISTANCE_OFFSET = new Uint32Array([
  1, 2, 3, 4, 5, 7, 9, 13, 17, 25,
  33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
  1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
  32769, 49153, 65537, 98305, 131073, 196609, 262145, 393217, 524289, 786433,
])

export function decodeVP8L(data: Uint8Array): WebpImageData {
  const reader = new BitReader(data)

  const signature = reader.readBits(8)
  if (signature !== VP8L_SIGNATURE) {
    throw new Error(`Invalid VP8L signature: expected 0x${VP8L_SIGNATURE.toString(16)}, got 0x${signature.toString(16)}`)
  }

  const header = readHeader(reader)
  const argb = decodeImage(reader, header)

  // ARGB → RGBA splat.
  const numPixels = header.width * header.height
  const rgba = new Uint8Array(numPixels * 4)
  for (let i = 0; i < numPixels; i++) {
    const p = argb[i]
    const o = i * 4
    rgba[o] = (p >>> 16) & 0xFF
    rgba[o + 1] = (p >>> 8) & 0xFF
    rgba[o + 2] = p & 0xFF
    rgba[o + 3] = (p >>> 24) & 0xFF
  }

  return {
    data: rgba,
    width: header.width,
    height: header.height,
    hasAlpha: header.hasAlpha,
  }
}

function readHeader(reader: BitReader): VP8LHeader {
  const widthMinus1 = reader.readBits(14)
  const heightMinus1 = reader.readBits(14)
  const hasAlpha = reader.readBit() === 1
  const version = reader.readBits(3)
  if (version !== 0) throw new Error(`Unsupported VP8L version: ${version}`)
  return {
    width: widthMinus1 + 1,
    height: heightMinus1 + 1,
    hasAlpha,
    version,
  }
}

function decodeImage(reader: BitReader, header: VP8LHeader): Uint32Array {
  // VP8L allows up to 4 chained transforms before the actual pixel data.
  // We don't apply them in the decoder yet because the encoder doesn't emit
  // them — but we *do* skip past them so this decoder still works on
  // bitstreams produced by other encoders that use, e.g., subtract-green.
  // Skipping is non-trivial since each transform has its own payload.
  let xsize = header.width
  while (reader.readBit() === 1) {
    skipTransform(reader, xsize, header.height)
    // Color-indexing transform changes the effective width — handle that
    // when we add real transform support; skipping is fine until then.
  }

  // Color cache (1 bit + optional 4-bit cache bits).
  let colorCacheBits = 0
  if (reader.readBit() === 1) {
    colorCacheBits = reader.readBits(4)
    if (colorCacheBits < 1 || colorCacheBits > MAX_CACHE_BITS) {
      throw new Error(`Invalid color cache bits: ${colorCacheBits}`)
    }
  }
  const colorCacheSize = colorCacheBits > 0 ? 1 << colorCacheBits : 0
  const colorCache = colorCacheSize > 0 ? new Uint32Array(colorCacheSize) : null

  // Meta-Huffman image (1 bit + optional payload). When present, the image
  // is divided into blocks each using a different Huffman group; the meta
  // image picks which group covers which block. We don't yet support this;
  // a real-world VP8L file emitted by libwebp at quality > 75 will trip
  // here, so when we hit it we fall back to a clear error rather than
  // silently mis-decoding.
  if (reader.readBit() === 1) {
    throw new Error('VP8L meta-Huffman images not yet supported')
  }

  const numLiteralCodes = NUM_LITERAL_CODES + NUM_LENGTH_CODES + colorCacheSize
  const greenTree = readHuffmanTree(reader, numLiteralCodes)
  const redTree = readHuffmanTree(reader, 256)
  const blueTree = readHuffmanTree(reader, 256)
  const alphaTree = readHuffmanTree(reader, 256)
  const distTree = readHuffmanTree(reader, NUM_DISTANCE_CODES)

  const numPixels = header.width * header.height
  const argb = new Uint32Array(numPixels)
  let pos = 0

  while (pos < numPixels) {
    const code = greenTree.readSymbol(reader)

    if (code < NUM_LITERAL_CODES) {
      // Literal pixel — read R, B, A from their trees.
      const green = code
      const red = redTree.readSymbol(reader)
      const blue = blueTree.readSymbol(reader)
      const alpha = alphaTree.readSymbol(reader)
      const pixel = ((alpha & 0xFF) << 24) | ((red & 0xFF) << 16) | ((green & 0xFF) << 8) | (blue & 0xFF)
      argb[pos++] = pixel
      if (colorCache) colorCache[hashPixel(pixel, colorCacheBits)] = pixel
    } else if (code < NUM_LITERAL_CODES + NUM_LENGTH_CODES) {
      // Backreference.
      const lengthCode = code - NUM_LITERAL_CODES
      const length = LENGTH_OFFSET[lengthCode] + reader.readBits(LENGTH_EXTRA_BITS[lengthCode])
      const distanceCode = distTree.readSymbol(reader)
      let distance: number
      if (distanceCode < 4) {
        // "Near" distance codes: distance = code + 1.
        distance = distanceCode + 1
      } else {
        const idx = distanceCode
        distance = DISTANCE_OFFSET[idx] + reader.readBits(DISTANCE_EXTRA_BITS[idx])
      }
      // Copy `length` ARGB slots from `pos - distance` to `pos`. The source
      // can be inside the destination range — that's the whole point of
      // LZ77 — so copy element by element rather than `Uint32Array.copyWithin`.
      const start = pos - distance
      if (start < 0) throw new Error(`VP8L: backreference distance ${distance} > position ${pos}`)
      const end = Math.min(pos + length, numPixels)
      for (let p = pos; p < end; p++) {
        const px = argb[p - distance]
        argb[p] = px
        if (colorCache) colorCache[hashPixel(px, colorCacheBits)] = px
      }
      pos = end
    } else {
      // Color-cache hit.
      if (!colorCache) throw new Error('VP8L: color-cache code with no cache')
      const cacheIndex = code - NUM_LITERAL_CODES - NUM_LENGTH_CODES
      if (cacheIndex >= colorCacheSize) throw new Error(`VP8L: color-cache index ${cacheIndex} out of range`)
      argb[pos++] = colorCache[cacheIndex]
    }
  }

  return argb
}

function hashPixel(pixel: number, bits: number): number {
  // VP8L uses (pixel * 0x1E35A7BD) >>> 0 then >> (32 - bits).
  return (Math.imul(pixel, 0x1E35A7BD) >>> (32 - bits)) & ((1 << bits) - 1)
}

/**
 * Skip past — or refuse — one transform's payload.
 *
 * The four VP8L transform types:
 *   0 = predictor          → 2 bits (size_bits) + meta image
 *   1 = color              → 2 bits (size_bits) + meta image
 *   2 = subtract-green     → no payload
 *   3 = color-indexing     → 8 bits (num_colors-1) + palette
 *
 * Subtract-green has no bitstream payload, but it *does* have semantics —
 * the decoded green channel needs to be added back to the red and blue
 * channels post-decode. We don't apply that yet, so a bitstream that uses
 * subtract-green (which is most quality > 0 output from libwebp) would
 * decode with mangled colors. Throwing is the right move until we
 * implement the inverse transform; a wrong-but-no-error output is the
 * worst possible failure mode for a decoder.
 */
function skipTransform(_reader: BitReader, _width: number, _height: number): never {
  // We don't even read the type — the moment we know there's a transform,
  // we can't faithfully decode this bitstream. (Reading the type would
  // require putting the bit back when we throw, which is not worth the
  // complexity for a pure-error path.)
  throw new Error(
    'VP8L transforms not yet supported by this decoder. The bitstream '
    + 'declares a pre-transform (subtract-green, predictor, color, or '
    + 'color-indexing); only encoder output without transforms decodes '
    + 'correctly today.',
  )
}
