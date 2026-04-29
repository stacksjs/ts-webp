import type { VP8LHeader, WebpImageData } from '../types'
import { BitReader } from '../bitreader'
import { DISTANCE_EXTRA_BITS, distanceFromCode, NUM_DISTANCE_CODES } from './distance'
import { HuffmanTree, readHuffmanTree } from './huffman'
import { LENGTH_EXTRA_BITS, lengthFromCode, NUM_LENGTH_CODES } from './length'

/**
 * VP8L (lossless WebP) decoder.
 *
 * Mirrors the encoder bit-for-bit and additionally handles bitstreams that
 * use the optional pre-transforms and color cache.
 *
 * Pipeline:
 *
 *   1. Read the 1-byte signature + 28-bit header.
 *   2. Read any number of pre-transforms; record them in order so we can
 *      reverse them at the end.
 *   3. Read color-cache flag (+ optional 4-bit cache_bits) and
 *      meta-Huffman flag (latter currently unsupported).
 *   4. Read the 5 Huffman trees.
 *   5. Decode the prefix-coded image into an ARGB Uint32Array, handling
 *      literal / backref / cache-hit codes.
 *   6. Reverse transforms in reverse order.
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
  // ── Read the transform chain ──
  // VP8L allows up to 4 chained transforms before pixel data. We only
  // implement subtract-green; the others throw so a caller doesn't get
  // silently mis-decoded pixels.
  const transforms: number[] = []
  while (reader.readBit() === 1) {
    if (transforms.length >= 4) throw new Error('VP8L: more than 4 transforms')
    const type = reader.readBits(2)
    if (type === TRANSFORM_SUBTRACT_GREEN) {
      transforms.push(type)
      // No payload.
    } else {
      const name = type === TRANSFORM_PREDICTOR ? 'predictor'
        : type === TRANSFORM_COLOR ? 'color'
          : type === TRANSFORM_COLOR_INDEXING ? 'color-indexing'
            : `unknown(${type})`
      throw new Error(`VP8L: ${name} transform not yet supported`)
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

  // ── Meta-Huffman flag ──
  if (reader.readBit() === 1) {
    throw new Error('VP8L meta-Huffman images not yet supported')
  }

  // ── 5 Huffman trees ──
  const numLiteralCodes = NUM_LITERAL_CODES + NUM_LENGTH_CODES + colorCacheSize
  const greenTree = readHuffmanTree(reader, numLiteralCodes)
  const redTree = readHuffmanTree(reader, 256)
  const blueTree = readHuffmanTree(reader, 256)
  const alphaTree = readHuffmanTree(reader, 256)
  const distTree = readHuffmanTree(reader, NUM_DISTANCE_CODES)

  // ── Decode pixels ──
  const numPixels = header.width * header.height
  const argb = new Uint32Array(numPixels)
  let pos = 0

  while (pos < numPixels) {
    const code = greenTree.readSymbol(reader)

    if (code < NUM_LITERAL_CODES) {
      // Literal pixel.
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
      // Backreference.
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
      // Color-cache hit.
      if (!colorCache) throw new Error('VP8L: color-cache code with no cache')
      const cacheIndex = code - NUM_LITERAL_CODES - NUM_LENGTH_CODES
      if (cacheIndex >= colorCacheSize) throw new Error(`VP8L: color-cache index ${cacheIndex} out of range`)
      argb[pos++] = colorCache[cacheIndex]
    }
  }

  // ── Reverse transforms (in reverse insertion order) ──
  for (let t = transforms.length - 1; t >= 0; t--) {
    if (transforms[t] === TRANSFORM_SUBTRACT_GREEN) inverseSubtractGreen(argb)
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
