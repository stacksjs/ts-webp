/**
 * ALPH chunk decoder — pulls a per-pixel alpha plane out of a
 * `VP8X + ALPH + VP8` container so VP8 (lossy) WebP files with
 * transparency decode correctly.
 *
 * Layout (RFC 9649 §9.1 + libwebp `alpha_dec.c`):
 *
 *   Byte 0 — header byte:
 *     bits 0..1  method        (0 = uncompressed, 1 = lossless VP8L)
 *     bits 2..3  filter         (0 = none, 1 = horizontal, 2 = vertical, 3 = gradient)
 *     bits 4..5  pre_processing (0 = none, 1 = level reduction; we don't apply)
 *     bits 6..7  reserved (must be 0)
 *
 *   Bytes 1..  — alpha data:
 *     • method=0: raw width*height bytes of filter-encoded alpha values
 *     • method=1: a complete VP8L bitstream whose decoded GREEN channel
 *       holds the filter-encoded alpha values (libwebp's "ExtractAlphaRows"
 *       grabs only the green plane after the inverse VP8L transforms run)
 *
 * After extracting the raw alpha bytes the per-row filter is unwound so
 * each pixel is reconstructed from its preceding neighbours.
 */
import { decodeVP8LImageStream } from './vp8l/decoder'

/**
 * Decode an ALPH chunk into a per-pixel alpha plane (one byte per
 * pixel, row-major). `width` and `height` come from the surrounding
 * VP8X / VP8 frame and aren't carried in the chunk itself.
 */
export function decodeAlpha(chunkData: Uint8Array, width: number, height: number): Uint8Array {
  if (chunkData.length < 1) {
    throw new Error('ts-webp: ALPH chunk shorter than 1-byte header')
  }
  const header = chunkData[0]
  const method = header & 0x03
  const filter = (header >> 2) & 0x03
  const preProc = (header >> 4) & 0x03
  const reserved = (header >> 6) & 0x03
  if (reserved !== 0) {
    throw new Error(`ts-webp: ALPH header reserved bits must be 0 (got ${reserved})`)
  }
  if (method !== 0 && method !== 1) {
    throw new Error(`ts-webp: ALPH unsupported compression method ${method}`)
  }

  const filtered = method === 0
    ? decodeUncompressed(chunkData.subarray(1), width, height)
    : decodeLossless(chunkData.subarray(1), width, height)

  // Unfilter row-by-row.
  return unfilterAlpha(filtered, width, height, filter, preProc)
}

function decodeUncompressed(data: Uint8Array, width: number, height: number): Uint8Array {
  const need = width * height
  if (data.length < need) {
    throw new Error(`ts-webp: ALPH uncompressed payload ${data.length} < ${need}`)
  }
  return data.subarray(0, need)
}

/**
 * The lossless path passes the bytes after the 1-byte header to a
 * standard VP8L decoder, then keeps only the green channel — that's
 * libwebp's `ExtractAlphaRows`. Wrapping the bitstream in a synthetic
 * RIFF/WEBP/VP8L container lets us reuse the existing `decodeVP8L`
 * entry point untouched.
 */
function decodeLossless(data: Uint8Array, width: number, height: number): Uint8Array {
  // ALPH lossless data is the "image stream" portion of a VP8L
  // bitstream (no signature/dimensions header — the surrounding
  // VP8X/VP8 container carries those). The decoded ARGB pixels carry
  // the filter-encoded alpha values in their green channel.
  const argb = decodeVP8LImageStream(data, width, height)
  const out = new Uint8Array(width * height)
  for (let i = 0; i < width * height; i++) {
    out[i] = (argb[i] >>> 8) & 0xFF // green channel
  }
  return out
}

function unfilterAlpha(
  src: Uint8Array, width: number, height: number,
  filter: number, _preProc: number,
): Uint8Array {
  // The output buffer is independent of the source so callers can safely
  // hold onto a subarray of the chunk if they want.
  const out = new Uint8Array(width * height)
  for (let y = 0; y < height; y++) {
    const rowOff = y * width
    const prevOff = y > 0 ? (y - 1) * width : -1
    unfilterRow(src, rowOff, prevOff >= 0 ? out : null, prevOff, out, rowOff, width, filter)
  }
  return out
}

/**
 * Unfilter a single row. `prev` is the previously-reconstructed row
 * (or null for row 0). Mirrors libwebp's `WebPUnfilters[filter]`.
 */
function unfilterRow(
  inBuf: Uint8Array, inOff: number,
  prev: Uint8Array | null, prevOff: number,
  outBuf: Uint8Array, outOff: number,
  width: number, filter: number,
): void {
  switch (filter) {
    case 0: // none
      for (let i = 0; i < width; i++) outBuf[outOff + i] = inBuf[inOff + i]
      return
    case 1: { // horizontal — pred = previous pixel in same row (or 0 for col 0)
      let pred = prev !== null ? prev[prevOff] : 0
      for (let i = 0; i < width; i++) {
        outBuf[outOff + i] = (pred + inBuf[inOff + i]) & 0xFF
        pred = outBuf[outOff + i]
      }
      return
    }
    case 2: // vertical — pred = pixel directly above (falls back to horizontal on row 0)
      if (prev === null) {
        unfilterRow(inBuf, inOff, null, 0, outBuf, outOff, width, 1)
        return
      }
      for (let i = 0; i < width; i++) {
        outBuf[outOff + i] = (prev[prevOff + i] + inBuf[inOff + i]) & 0xFF
      }
      return
    case 3: { // gradient — pred = clamp(left + top - top_left, 0..255)
      if (prev === null) {
        unfilterRow(inBuf, inOff, null, 0, outBuf, outOff, width, 1)
        return
      }
      let topLeft = prev[prevOff]
      let left = topLeft
      for (let i = 0; i < width; i++) {
        const top = prev[prevOff + i]
        const g = left + top - topLeft
        const pred = g < 0 ? 0 : g > 255 ? 255 : g
        left = (inBuf[inOff + i] + pred) & 0xFF
        outBuf[outOff + i] = left
        topLeft = top
      }
    }
  }
}
