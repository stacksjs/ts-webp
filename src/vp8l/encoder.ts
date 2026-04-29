import type { WebpEncodeOptions, WebpImageData } from '../types'
import { BitWriter } from '../bitreader'
import { buildCodeLengths, lengthsToCodes, reverseBits, writeHuffmanTree } from './huffman'

/**
 * VP8L (lossless WebP) encoder.
 *
 * This is a *correct* but not yet *small* encoder: it produces spec-compliant
 * VP8L bitstreams that round-trip with our decoder (and with libwebp), but
 * skips the optional pre-transforms (subtract-green, predictor, color, color
 * indexing) that real production encoders use to shave 30-60 % off the
 * output size on natural images. Adding those is straightforward — each is
 * an independent pass over the pixel buffer with a 1-bit flag in the
 * bitstream — and is staged in subtract-green / LZ77 / color-cache work.
 *
 * The bitstream layout we emit:
 *
 *   1 byte   VP8L signature (0x2F)
 *   28 bits  header: width-1 (14) | height-1 (14) | alpha (1) | version (3)
 *   1 bit    transform-present = 0
 *   1 bit    color-cache-present = 0
 *   1 bit    meta-Huffman-image-present = 0
 *   5 trees  literal/length, red, blue, alpha, distance
 *   N codes  per-pixel: G, R, B, A as literals
 *
 * Reference: WebP Lossless Bitstream Specification.
 */

/** VP8L magic byte that prefixes every lossless image. */
const VP8L_SIGNATURE = 0x2F

/** Number of literal-only codes in the green/length alphabet. */
const NUM_LITERAL_CODES = 256
/** Number of length codes in the green/length alphabet. */
const NUM_LENGTH_CODES = 24
/** Total green/length symbols *before* color-cache codes are appended. */
const GREEN_BASE = NUM_LITERAL_CODES + NUM_LENGTH_CODES
/** Number of distance codes. */
const NUM_DISTANCE_CODES = 40

export function encodeVP8L(
  imageData: WebpImageData,
  _options: WebpEncodeOptions = {},
): Uint8Array {
  const { data, width, height } = imageData
  if (width <= 0 || height <= 0) throw new Error('VP8L: width and height must be positive')
  if (width > 16384 || height > 16384) throw new Error('VP8L: dimensions cap at 16384')
  if (data.length < width * height * 4) throw new Error('VP8L: pixel buffer is shorter than width × height × 4')

  // Estimate a sensible starting buffer so encode doesn't repeatedly grow.
  // Worst-case literal output is roughly 4 bytes per pixel + tree overhead.
  const writer = new BitWriter(width * height * 4 + 1024)

  // ── 1-byte signature ──
  writer.writeBits(VP8L_SIGNATURE, 8)

  // ── 28-bit header ──
  writer.writeBits(width - 1, 14)
  writer.writeBits(height - 1, 14)
  writer.writeBit(imageData.hasAlpha ? 1 : 0) // alpha hint
  writer.writeBits(0, 3) // version (must be 0)

  // ── No transforms (1 bit = 0) ──
  writer.writeBit(0)

  // ── No color cache (1 bit = 0) ──
  writer.writeBit(0)

  // ── No meta-Huffman image (1 bit = 0) → single Huffman group covers
  //    the entire image. The spec calls this "is_meta_huffman = 0", which
  //    confusingly is the flag inside the *prefix-coded image*; the
  //    top-level "ColorCacheInfo" / "HuffmanCodes" sequence reads the
  //    color-cache bit (above) and then *another* bit here for the
  //    meta-Huffman image presence. Some references combine them. We
  //    explicitly write zero so the decoder skips reading the meta image.
  writer.writeBit(0)

  // Build per-channel frequency histograms across the whole image.
  const greenFreq = new Uint32Array(GREEN_BASE)
  const redFreq = new Uint32Array(256)
  const blueFreq = new Uint32Array(256)
  const alphaFreq = new Uint32Array(256)
  // Distance code frequencies — empty for now since we only emit literals,
  // but VP8L still requires a valid distance tree on the wire. Give it one
  // symbol so `buildCodeLengths` produces a length-1 tree we can encode
  // via the simple-code path.
  const distFreq = new Uint32Array(NUM_DISTANCE_CODES)
  distFreq[0] = 1

  const numPixels = width * height
  for (let i = 0; i < numPixels; i++) {
    const o = i * 4
    redFreq[data[o]]++
    greenFreq[data[o + 1]]++
    blueFreq[data[o + 2]]++
    alphaFreq[data[o + 3]]++
  }

  // ── Build & emit each Huffman tree ──
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

  // ── Emit pixel data ──
  // Per spec, the order on the wire is G, R, B, A — green carries the
  // length codes for backreferences, so it's read first; the others
  // follow only when green is a literal.
  const greenCodes = lengthsToCodes(greenLen)
  const redCodes = lengthsToCodes(redLen)
  const blueCodes = lengthsToCodes(blueLen)
  const alphaCodes = lengthsToCodes(alphaLen)

  for (let i = 0; i < numPixels; i++) {
    const o = i * 4
    const r = data[o]
    const g = data[o + 1]
    const b = data[o + 2]
    const a = data[o + 3]
    writeSymbol(writer, g, greenLen[g], greenCodes[g])
    writeSymbol(writer, r, redLen[r], redCodes[r])
    writeSymbol(writer, b, blueLen[b], blueCodes[b])
    writeSymbol(writer, a, alphaLen[a], alphaCodes[a])
  }

  return writer.getBuffer()
}

/**
 * Emit a single Huffman-coded symbol. Reverses the canonical code into
 * wire order (LSB-first within byte) before writing.
 */
function writeSymbol(writer: BitWriter, _sym: number, len: number, code: number): void {
  if (len === 0) {
    // Should not happen for symbols that actually appear in the image —
    // their frequency is non-zero so `buildCodeLengths` gave them ≥ 1 bit.
    // If it does, the encoder has a bug; fail loudly so it's caught in tests.
    throw new Error('writeSymbol: zero-length code for an emitted symbol')
  }
  writer.writeBits(reverseBits(code, len), len)
}
