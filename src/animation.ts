import type { WebpAnimation, WebpAnimationFrame } from './types'
import { parseRiff } from './riff'
import { decodeVP8L } from './vp8l/decoder'

/**
 * Decode an animated WebP container into its constituent frames + the
 * canvas-level metadata needed to play them back (loop count, background
 * colour, per-frame timing/positioning/composition rules).
 *
 * Container layout (extended-format WebP with animation):
 *
 *   RIFF / WEBP wrapper
 *     VP8X chunk      — extended-format flags + canvas dimensions
 *     ANIM chunk      — 4-byte BG color (ARGB) + 2-byte loop count
 *     ANMF chunk      — frame N: 16-byte header + frame's own chunks
 *     ANMF chunk      — frame N+1
 *     …
 *
 * Each ANMF chunk's payload is a header followed by a sequence of
 * sub-chunks (typically a single VP8L; possibly a VP8X+ALPH+VP8 trio
 * for lossy frames). We dispatch each frame's payload into the
 * lossless decoder.
 *
 * Reference: WebP Container Specification, "Animation".
 */

/** Decode `buffer` as an animated WebP. Throws if the file isn't animated. */
export function decodeAnimation(buffer: Uint8Array | ArrayBuffer): WebpAnimation {
  const data = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer
  const chunks = parseRiff(data)

  const vp8x = chunks.find(c => c.fourCC === 'VP8X')
  if (!vp8x) {
    throw new Error('ts-webp: not an extended-format WebP — no VP8X chunk')
  }

  // VP8X flags (byte 0): bit 1 (0x02) = animation, bit 4 (0x10) = alpha,
  // bit 5 (0x20) = ICCP, bit 3 (0x08) = EXIF, bit 2 (0x04) = XMP.
  const flags = vp8x.data[0]
  if ((flags & 0x02) === 0) {
    throw new Error('ts-webp: not animated — VP8X animation flag is clear')
  }

  // Canvas dimensions are 24-bit little-endian, stored as `dim - 1`.
  const width = readUint24LE(vp8x.data, 4) + 1
  const height = readUint24LE(vp8x.data, 7) + 1

  // ANIM chunk: 4-byte BG ARGB + 2-byte loop count.
  const anim = chunks.find(c => c.fourCC === 'ANIM')
  let backgroundColor = 0
  let loopCount = 0
  if (anim && anim.data.length >= 6) {
    // BG color stored as B, G, R, A in spec order (yes, alpha last).
    backgroundColor = (anim.data[3] << 24) | (anim.data[2] << 16) | (anim.data[1] << 8) | anim.data[0]
    loopCount = anim.data[4] | (anim.data[5] << 8)
    backgroundColor = backgroundColor >>> 0
  }

  // Each ANMF is one frame. Spec keeps them in display order.
  const frames: WebpAnimationFrame[] = []
  for (const chunk of chunks) {
    if (chunk.fourCC !== 'ANMF') continue
    if (chunk.data.length < 16) {
      throw new Error('ts-webp: malformed ANMF chunk (header < 16 bytes)')
    }
    const fx = readUint24LE(chunk.data, 0) * 2
    const fy = readUint24LE(chunk.data, 3) * 2
    const fwidth = readUint24LE(chunk.data, 6) + 1
    const fheight = readUint24LE(chunk.data, 9) + 1
    const fduration = readUint24LE(chunk.data, 12)
    const ffFlags = chunk.data[15]
    // Bit 0: blending (0 = use alpha-blending, 1 = no blending → replace)
    // Bit 1: dispose method (0 = none, 1 = background)
    const blend = (ffFlags & 0x02) !== 0 ? 'replace' : 'overlay'
    const dispose = (ffFlags & 0x01) !== 0 ? 'background' : 'none'

    // Frame payload is a fragment of chunks (no outer RIFF wrapper).
    // We re-parse it directly from offset 16 of the ANMF data.
    const frameBytes = chunk.data.subarray(16)
    const image = decodeFramePayload(frameBytes, fwidth, fheight)
    frames.push({ image, x: fx, y: fy, duration: fduration, blend, dispose })
  }

  return { width, height, loopCount, backgroundColor, frames }
}

/**
 * Read a 24-bit little-endian unsigned integer from `data` at `offset`.
 * VP8X fields use this packing; built-in DataView has no 24-bit reader.
 */
function readUint24LE(data: Uint8Array, offset: number): number {
  return data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16)
}

/**
 * Decode the chunk-sequence payload that follows an ANMF header.
 *
 * Per the container spec, this isn't a full RIFF/WEBP — it's just a
 * sequence of `(fourCC, size, data)` chunk records. The frame's pixels
 * live in either a single `VP8L` chunk (lossless) or a `VP8 ` plus
 * optional `ALPH` (lossy + alpha) — we currently only handle the
 * VP8L path; lossy frames throw via the underlying VP8 decoder.
 */
function decodeFramePayload(payload: Uint8Array, expectedWidth: number, expectedHeight: number): import('./types').WebpImageData {
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)
  let offset = 0
  let vp8lData: Uint8Array | null = null

  while (offset + 8 <= payload.length) {
    const fourCC = String.fromCharCode(
      payload[offset],
      payload[offset + 1],
      payload[offset + 2],
      payload[offset + 3],
    )
    const size = view.getUint32(offset + 4, true)
    const dataStart = offset + 8
    const dataEnd = Math.min(dataStart + size, payload.length)
    if (fourCC === 'VP8L') vp8lData = payload.subarray(dataStart, dataEnd)
    // Pad to even boundary.
    const padded = size + (size & 1)
    offset = dataStart + padded
  }

  if (!vp8lData) {
    throw new Error('ts-webp: animation frame has no VP8L chunk (lossy frames not supported)')
  }
  const decoded = decodeVP8L(vp8lData)
  if (decoded.width !== expectedWidth || decoded.height !== expectedHeight) {
    throw new Error(
      `ts-webp: ANMF declared frame ${expectedWidth}×${expectedHeight}`
      + ` but VP8L decoded ${decoded.width}×${decoded.height}`,
    )
  }
  return decoded
}
