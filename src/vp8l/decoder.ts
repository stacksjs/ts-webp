import type { VP8LHeader, WebpImageData } from '../types'
import { BitReader } from '../bitreader'
import { DISTANCE_EXTRA_BITS, distanceFromCode, NUM_DISTANCE_CODES } from './distance'
import { HuffmanTree, readHuffmanTree } from './huffman'
import { LENGTH_EXTRA_BITS, lengthFromCode, NUM_LENGTH_CODES } from './length'
import { inverseColorTransform } from './color'
import { deltaDecodePalette, inverseColorIndexTransform, packingFactor } from './color-index'
import { inversePredictorTransform } from './predictor'

/**
 * VP8L (lossless WebP) decoder.
 *
 * Mirrors the encoder bit-for-bit. Pipeline:
 *
 *   1. Read the 1-byte signature + 28-bit header.
 *   2. Read each pre-transform's metadata in bitstream order. Predictor
 *      and color transforms each carry a recursively-encoded sub-image
 *      that we read here too.
 *   3. Read color-cache flag (+ optional 4-bit cache_bits) and
 *      meta-Huffman flag (latter unsupported).
 *   4. Read the 5 Huffman trees.
 *   5. Decode the prefix-coded image into an ARGB Uint32Array, handling
 *      literal / backref / cache-hit codes.
 *   6. Apply transform inverses *in bitstream order*. The encoder stored
 *      them in reverse application order, so first-read = first-inverse,
 *      which unwinds the encoder's stack.
 *   7. Convert ARGB → RGBA and return.
 *
 * Reference: WebP Lossless Bitstream Specification.
 */

const VP8L_SIGNATURE = 0x2F
const NUM_LITERAL_CODES = 256
const MAX_CACHE_BITS = 11

const VP8L_HASH_MUL = 0x1E35A7BD

const TRANSFORM_PREDICTOR = 0
const TRANSFORM_COLOR = 1
const TRANSFORM_SUBTRACT_GREEN = 2
const TRANSFORM_COLOR_INDEXING = 3

/** A Huffman group is the 5-tree set used for one block of the image. */
interface HuffmanGroup {
  green: HuffmanTree
  red: HuffmanTree
  blue: HuffmanTree
  alpha: HuffmanTree
  distance: HuffmanTree
}

/** Recorded transform with any payload it needs at inverse-apply time. */
interface TransformRecord {
  type: number
  bits?: number
  modeImage?: Uint32Array
  modeWidth?: number
  modeHeight?: number
  /** For color-indexing: the post-delta-decode palette. */
  palette?: Uint32Array
  /** For color-indexing: the encoded width seen by the pixel decoder. */
  encodedWidth?: number
}

export function decodeVP8L(data: Uint8Array): WebpImageData {
  const reader = new BitReader(data)

  const signature = reader.readBits(8)
  if (signature !== VP8L_SIGNATURE) {
    throw new Error(`Invalid VP8L signature: expected 0x${VP8L_SIGNATURE.toString(16)}, got 0x${signature.toString(16)}`)
  }

  const header = readHeader(reader)
  const argb = decodeImage(reader, header)

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
  // ── Read the transform chain (max 4 entries per spec) ──
  // `effectiveWidth` tracks the width seen by transforms that come *after*
  // a color-indexing transform — color-indexing reduces the encoded image
  // width, and downstream predictor mode-image sizes are derived from
  // the reduced width, not the original.
  const transforms: TransformRecord[] = []
  let effectiveWidth = header.width
  while (reader.readBit() === 1) {
    if (transforms.length >= 4) throw new Error('VP8L: more than 4 transforms')
    const type = reader.readBits(2)
    if (type === TRANSFORM_SUBTRACT_GREEN) {
      transforms.push({ type })
    } else if (type === TRANSFORM_PREDICTOR) {
      // Predictor: 3 bits (size_bits - 2) then a sub-image of mode-per-block.
      const bits = reader.readBits(3) + 2
      const modeWidth = (effectiveWidth + (1 << bits) - 1) >>> bits
      const modeHeight = (header.height + (1 << bits) - 1) >>> bits
      const modeImage = readSubImage(reader, modeWidth, modeHeight)
      transforms.push({ type, bits, modeImage, modeWidth, modeHeight })
    } else if (type === TRANSFORM_COLOR) {
      // Color: 3-bit (size_bits-2) + a per-block coefficient sub-image.
      const bits = reader.readBits(3) + 2
      const ctWidth = (effectiveWidth + (1 << bits) - 1) >>> bits
      const ctHeight = (header.height + (1 << bits) - 1) >>> bits
      const ctImage = readSubImage(reader, ctWidth, ctHeight)
      transforms.push({ type, bits, modeImage: ctImage, modeWidth: ctWidth, modeHeight: ctHeight })
    } else if (type === TRANSFORM_COLOR_INDEXING) {
      // Color-indexing: 8-bit palette_size-1 + 1-row sub-image with the
      // delta-encoded palette. After this transform the decoded pixel
      // stream operates at a reduced width (palette ≤ 16 packs multiple
      // indices per output pixel).
      const paletteSize = reader.readBits(8) + 1
      const deltaPalette = readSubImage(reader, paletteSize, 1)
      const palette = deltaDecodePalette(deltaPalette)
      const factor = packingFactor(paletteSize)
      const encodedWidth = (effectiveWidth + factor - 1) / factor | 0
      effectiveWidth = encodedWidth
      transforms.push({ type, palette, encodedWidth })
    } else {
      throw new Error(`VP8L: unknown transform type ${type}`)
    }
  }

  // ── Color cache flag ──
  let colorCacheBits = 0
  if (reader.readBit() === 1) {
    colorCacheBits = reader.readBits(4)
    if (colorCacheBits < 1 || colorCacheBits > MAX_CACHE_BITS) {
      throw new Error(`Invalid color cache bits: ${colorCacheBits}`)
    }
  }
  const colorCacheSize = colorCacheBits > 0 ? 1 << colorCacheBits : 0
  const colorCache = colorCacheSize > 0 ? new Uint32Array(colorCacheSize) : null

  // ── Meta-Huffman image ──
  // Real-world libwebp output at quality > 75 uses multiple Huffman
  // groups, with the per-pixel choice driven by a meta-image (one
  // pixel per `2^bits × 2^bits` source block). We support this end-to-end
  // on decode; our encoder always uses a single group.
  let metaHuffmanBits = 0
  let metaHuffmanImage: Uint32Array | null = null
  let metaHuffmanWidth = 0
  let numHuffmanGroups = 1
  if (reader.readBit() === 1) {
    metaHuffmanBits = reader.readBits(3) + 2
    metaHuffmanWidth = (effectiveWidth + (1 << metaHuffmanBits) - 1) >>> metaHuffmanBits
    const metaH = (header.height + (1 << metaHuffmanBits) - 1) >>> metaHuffmanBits
    metaHuffmanImage = readSubImage(reader, metaHuffmanWidth, metaH)
    // Group index lives in (red << 8) | green of each meta pixel; find max.
    let maxGroup = 0
    for (let i = 0; i < metaHuffmanImage.length; i++) {
      const px = metaHuffmanImage[i]
      const idx = ((px >>> 16) & 0xFF) * 256 + ((px >>> 8) & 0xFF)
      if (idx > maxGroup) maxGroup = idx
    }
    numHuffmanGroups = maxGroup + 1
  }

  // ── 5 trees per group ──
  const numLiteralCodes = NUM_LITERAL_CODES + NUM_LENGTH_CODES + colorCacheSize
  const groups: HuffmanGroup[] = []
  for (let g = 0; g < numHuffmanGroups; g++) {
    groups.push({
      green: readHuffmanTree(reader, numLiteralCodes),
      red: readHuffmanTree(reader, 256),
      blue: readHuffmanTree(reader, 256),
      alpha: readHuffmanTree(reader, 256),
      distance: readHuffmanTree(reader, NUM_DISTANCE_CODES),
    })
  }

  // ── Decode pixels ──
  // The pixel stream is sized at `effectiveWidth × header.height` — color-
  // indexing may have shrunk the working width below the original.
  let argb: Uint32Array
  if (metaHuffmanImage) {
    argb = decodePixelStreamMeta(
      reader,
      groups,
      effectiveWidth,
      header.height,
      metaHuffmanBits,
      metaHuffmanImage,
      metaHuffmanWidth,
      colorCacheBits,
      colorCache,
    )
  } else {
    const g = groups[0]
    argb = decodePixelStream(
      reader,
      g.green,
      g.red,
      g.blue,
      g.alpha,
      g.distance,
      effectiveWidth * header.height,
      colorCacheBits,
      colorCache,
    )
  }

  // ── Apply transform inverses in REVERSE bitstream order ──
  // Spec: transforms appear in the bitstream in their forward-application
  // order, so to unwind we walk from last to first. With encoder applying
  // SG → predictor, the bitstream is [SG, predictor] and the decoder
  // applies inverse-predictor first (un-doing the last-applied transform)
  // then inverse-SG. For color-indexing alone, the loop runs once.
  let currentWidth = effectiveWidth
  for (let i = transforms.length - 1; i >= 0; i--) {
    const t = transforms[i]
    if (t.type === TRANSFORM_SUBTRACT_GREEN) {
      inverseSubtractGreen(argb)
    } else if (t.type === TRANSFORM_PREDICTOR) {
      inversePredictorTransform(argb, currentWidth, header.height, t.bits!, t.modeImage!, t.modeWidth!)
    } else if (t.type === TRANSFORM_COLOR) {
      inverseColorTransform(argb, currentWidth, header.height, t.bits!, t.modeImage!, t.modeWidth!)
    } else if (t.type === TRANSFORM_COLOR_INDEXING) {
      // Inverse color-indexing un-packs to the *original* (pre-shrink)
      // width on this side of the transform stack, restoring full RGBA.
      const palette = t.palette!
      const factor = packingFactor(palette.length)
      const originalWidth = currentWidth * factor // approximate; clamp below
      const targetWidth = Math.min(originalWidth, header.width)
      argb = inverseColorIndexTransform(argb, currentWidth, targetWidth, header.height, palette)
      currentWidth = targetWidth
    }
  }

  return argb
}

/**
 * Read a sub-image — used by the predictor + color transforms for their
 * meta images. Sub-images have no header and no nested transforms; just
 * a transform-present=0 bit, color-cache flag, meta-Huffman flag, the
 * 5 Huffman trees, and the prefix-coded pixel stream.
 */
function readSubImage(reader: BitReader, width: number, height: number): Uint32Array {
  // Sub-image must have transform-present=0. Reading any 1 here would
  // mean the encoder nested transforms inside a sub-image, which is
  // legal per spec but our encoder never produces it — and supporting
  // it on decode wouldn't help any real workload.
  if (reader.readBit() === 1) {
    throw new Error('VP8L: nested transforms inside sub-image not supported')
  }

  let colorCacheBits = 0
  if (reader.readBit() === 1) {
    colorCacheBits = reader.readBits(4)
    if (colorCacheBits < 1 || colorCacheBits > MAX_CACHE_BITS) {
      throw new Error(`Invalid sub-image color cache bits: ${colorCacheBits}`)
    }
  }
  const colorCacheSize = colorCacheBits > 0 ? 1 << colorCacheBits : 0
  const colorCache = colorCacheSize > 0 ? new Uint32Array(colorCacheSize) : null

  if (reader.readBit() === 1) {
    throw new Error('VP8L: meta-Huffman in sub-image not supported')
  }

  const numLiteralCodes = NUM_LITERAL_CODES + NUM_LENGTH_CODES + colorCacheSize
  const greenTree = readHuffmanTree(reader, numLiteralCodes)
  const redTree = readHuffmanTree(reader, 256)
  const blueTree = readHuffmanTree(reader, 256)
  const alphaTree = readHuffmanTree(reader, 256)
  const distTree = readHuffmanTree(reader, NUM_DISTANCE_CODES)

  return decodePixelStream(
    reader,
    greenTree,
    redTree,
    blueTree,
    alphaTree,
    distTree,
    width * height,
    colorCacheBits,
    colorCache,
  )
}

/**
 * Decode a prefix-coded pixel stream of `numPixels` ARGB words, handling
 * literal / backref / color-cache codes. Shared between the main image
 * and any nested sub-image — same bitstream format.
 */
function decodePixelStream(
  reader: BitReader,
  greenTree: HuffmanTree,
  redTree: HuffmanTree,
  blueTree: HuffmanTree,
  alphaTree: HuffmanTree,
  distTree: HuffmanTree,
  numPixels: number,
  colorCacheBits: number,
  colorCache: Uint32Array | null,
): Uint32Array {
  const colorCacheSize = colorCache ? colorCache.length : 0
  const argb = new Uint32Array(numPixels)
  let pos = 0

  while (pos < numPixels) {
    const code = greenTree.readSymbol(reader)

    if (code < NUM_LITERAL_CODES) {
      const green = code
      const red = redTree.readSymbol(reader)
      const blue = blueTree.readSymbol(reader)
      const alpha = alphaTree.readSymbol(reader)
      const pixel = ((alpha & 0xFF) << 24) | ((red & 0xFF) << 16) | ((green & 0xFF) << 8) | (blue & 0xFF)
      argb[pos++] = pixel
      if (colorCache) {
        const slot = (Math.imul(pixel, VP8L_HASH_MUL) >>> (32 - colorCacheBits)) & (colorCacheSize - 1)
        colorCache[slot] = pixel
      }
    } else if (code < NUM_LITERAL_CODES + NUM_LENGTH_CODES) {
      const lengthCode = code - NUM_LITERAL_CODES
      const length = lengthFromCode(lengthCode, reader.readBits(LENGTH_EXTRA_BITS[lengthCode]))
      const distanceCode = distTree.readSymbol(reader)
      const distance = distanceFromCode(distanceCode, reader.readBits(DISTANCE_EXTRA_BITS[distanceCode]))
      const start = pos - distance
      if (start < 0) throw new Error(`VP8L: backreference distance ${distance} > position ${pos}`)
      const end = Math.min(pos + length, numPixels)
      for (let p = pos; p < end; p++) {
        const px = argb[p - distance]
        argb[p] = px
        if (colorCache) {
          const slot = (Math.imul(px, VP8L_HASH_MUL) >>> (32 - colorCacheBits)) & (colorCacheSize - 1)
          colorCache[slot] = px
        }
      }
      pos = end
    } else {
      if (!colorCache) throw new Error('VP8L: color-cache code with no cache')
      const cacheIndex = code - NUM_LITERAL_CODES - NUM_LENGTH_CODES
      if (cacheIndex >= colorCacheSize) throw new Error(`VP8L: color-cache index ${cacheIndex} out of range`)
      argb[pos++] = colorCache[cacheIndex]
    }
  }

  return argb
}

/**
 * Decode a prefix-coded pixel stream that uses a meta-Huffman image:
 * each `2^bits × 2^bits` block of the image picks one of `groups[]` to
 * read the next pixels. Otherwise identical to `decodePixelStream`.
 *
 * Group switches happen at *block boundaries*, but a backreference can
 * span blocks — when it does, the group in effect for the rest of the
 * backref is whatever was active when the run started. Cache updates
 * touch every reproduced pixel regardless of group.
 */
function decodePixelStreamMeta(
  reader: BitReader,
  groups: HuffmanGroup[],
  width: number,
  height: number,
  metaBits: number,
  metaImage: Uint32Array,
  metaWidth: number,
  colorCacheBits: number,
  colorCache: Uint32Array | null,
): Uint32Array {
  const colorCacheSize = colorCache ? colorCache.length : 0
  const numPixels = width * height
  const argb = new Uint32Array(numPixels)
  let pos = 0
  let lastGroupX = -1
  let lastGroupY = -1
  let group = groups[0]

  while (pos < numPixels) {
    const x = pos % width
    const y = (pos / width) | 0
    const bx = x >>> metaBits
    const by = y >>> metaBits
    if (bx !== lastGroupX || by !== lastGroupY) {
      lastGroupX = bx
      lastGroupY = by
      const meta = metaImage[by * metaWidth + bx]
      const gIdx = ((meta >>> 16) & 0xFF) * 256 + ((meta >>> 8) & 0xFF)
      if (gIdx >= groups.length) throw new Error(`VP8L: meta-Huffman group index ${gIdx} out of range`)
      group = groups[gIdx]
    }

    const code = group.green.readSymbol(reader)
    if (code < NUM_LITERAL_CODES) {
      const green = code
      const red = group.red.readSymbol(reader)
      const blue = group.blue.readSymbol(reader)
      const alpha = group.alpha.readSymbol(reader)
      const pixel = ((alpha & 0xFF) << 24) | ((red & 0xFF) << 16) | ((green & 0xFF) << 8) | (blue & 0xFF)
      argb[pos++] = pixel
      if (colorCache) {
        const slot = (Math.imul(pixel, VP8L_HASH_MUL) >>> (32 - colorCacheBits)) & (colorCacheSize - 1)
        colorCache[slot] = pixel
      }
    } else if (code < NUM_LITERAL_CODES + NUM_LENGTH_CODES) {
      const lengthCode = code - NUM_LITERAL_CODES
      const length = lengthFromCode(lengthCode, reader.readBits(LENGTH_EXTRA_BITS[lengthCode]))
      const distanceCode = group.distance.readSymbol(reader)
      const distance = distanceFromCode(distanceCode, reader.readBits(DISTANCE_EXTRA_BITS[distanceCode]))
      const start = pos - distance
      if (start < 0) throw new Error(`VP8L: backreference distance ${distance} > position ${pos}`)
      const end = Math.min(pos + length, numPixels)
      for (let p = pos; p < end; p++) {
        const px = argb[p - distance]
        argb[p] = px
        if (colorCache) {
          const slot = (Math.imul(px, VP8L_HASH_MUL) >>> (32 - colorCacheBits)) & (colorCacheSize - 1)
          colorCache[slot] = px
        }
      }
      pos = end
    } else {
      if (!colorCache) throw new Error('VP8L: color-cache code with no cache')
      const cacheIndex = code - NUM_LITERAL_CODES - NUM_LENGTH_CODES
      if (cacheIndex >= colorCacheSize) throw new Error(`VP8L: color-cache index ${cacheIndex} out of range`)
      argb[pos++] = colorCache[cacheIndex]
    }
  }

  return argb
}

function inverseSubtractGreen(argb: Uint32Array): void {
  for (let i = 0; i < argb.length; i++) {
    const px = argb[i]
    const a = px & 0xFF000000
    const r = (px >>> 16) & 0xFF
    const g = (px >>> 8) & 0xFF
    const b = px & 0xFF
    const r2 = (r + g) & 0xFF
    const b2 = (b + g) & 0xFF
    argb[i] = (a | (r2 << 16) | (g << 8) | b2) >>> 0
  }
}
