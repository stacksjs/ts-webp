/**
 * VP8L distance-code helpers.
 *
 * Distances in VP8L are LZ77 backreferences expressed as a single prefix
 * code 0..39 plus extra bits for codes ≥ 4. The code → distance mapping
 * is DEFLATE-style: each code represents a base distance with a fixed
 * number of extra bits to read.
 *
 * On TOP of that, the spec applies a 120-entry "plane code" remapping
 * (libwebp's `kCodeToPlane`): the first 120 raw distance values aren't
 * linear pixel distances at all but compact 2D `(xoffset, yoffset)`
 * tuples relative to the current pixel — so "the pixel directly above"
 * is a 1-2 bit code instead of `xsize`-bits. Raw values > 120 are linear
 * distances minus 120. The decoder converts via
 *   xoffset = 8 - (kCodeToPlane[raw - 1] & 0x0F)
 *   yoffset = kCodeToPlane[raw - 1] >> 4
 *   actual  = max(1, yoffset * xsize + xoffset)
 * (matching libwebp's `PlaneCodeToDistance`). The encoder inverts this
 * by mapping the linear distance back into a plane code when one of the
 * 120 entries lands within ±7 of the current pixel.
 *
 * Reference: WebP Lossless Bitstream Specification §8.2.
 */

/** Number of distance prefix codes in the VP8L distance alphabet. */
export const NUM_DISTANCE_CODES = 40

/**
 * Per-spec extra-bit count for distance prefix codes 0..39. The first 4
 * codes are direct (no extra bits); each pair afterwards doubles the
 * range covered.
 */
export const DISTANCE_EXTRA_BITS = new Uint8Array([
  0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
  9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
])

/**
 * Per-spec base distance for prefix codes 0..39. Code `c` covers
 * `[DISTANCE_OFFSET[c], DISTANCE_OFFSET[c] + (1 << DISTANCE_EXTRA_BITS[c]) - 1]`.
 */
export const DISTANCE_OFFSET = new Uint32Array([
  1, 2, 3, 4, 5, 7, 9, 13, 17, 25,
  33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
  1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
  32769, 49153, 65537, 98305, 131073, 196609, 262145, 393217, 524289, 786433,
])

/**
 * Largest distance representable in the alphabet. A backreference whose
 * linear distance exceeds this can't be encoded — the encoder must fall
 * back to literal-emit for that pixel.
 */
export const MAX_DISTANCE = DISTANCE_OFFSET[NUM_DISTANCE_CODES - 1]
  + (1 << DISTANCE_EXTRA_BITS[NUM_DISTANCE_CODES - 1]) - 1

/**
 * For decoding: given a prefix code 0..39 and the extra-bits *value*
 * already read, return the RAW distance value (the value before plane
 * remapping). The caller passes this through `planeCodeToDistance` to
 * recover the actual pixel distance. Splitting the two steps keeps the
 * Huffman/extra-bits decoding and the spatial mapping cleanly separable.
 */
export function distanceFromCode(code: number, extraBitsValue: number): number {
  if (code < 0 || code >= NUM_DISTANCE_CODES) {
    throw new Error(`distanceFromCode: code ${code} out of range`)
  }
  if (code < 4) return code + 1
  return DISTANCE_OFFSET[code] + extraBitsValue
}

/**
 * For encoding: convert a *raw* distance value (post-plane-remap) into
 * the prefix code + extra-bits payload. Returns `null` if the value is
 * out of range. Encoders that want to pick the most compact 2D offset
 * for a given linear distance should call `linearDistanceToRaw` first
 * to get a list of candidate raw values, score the prefix-encoded sizes,
 * and pass the winner here.
 */
export function distanceToCode(rawDistance: number): {
  code: number
  extraBits: number
  extraValue: number
} | null {
  if (rawDistance < 1 || rawDistance > MAX_DISTANCE) return null
  if (rawDistance <= 4) return { code: rawDistance - 1, extraBits: 0, extraValue: 0 }
  // Binary search would be marginally faster, but the alphabet is 40
  // entries — linear scan is fine and easier to audit.
  for (let code = 4; code < NUM_DISTANCE_CODES; code++) {
    const base = DISTANCE_OFFSET[code]
    const extraBits = DISTANCE_EXTRA_BITS[code]
    if (rawDistance >= base && rawDistance < base + (1 << extraBits)) {
      return { code, extraBits, extraValue: rawDistance - base }
    }
  }
  return null
}

/**
 * The 120-entry "plane code" table from the VP8L spec. Each byte encodes
 * a packed (yoffset, xoffset) pair — see `planeCodeToDistance` for the
 * unpacking. Order matches libwebp's `kCodeToPlane24` in `huffman_utils.h`.
 */
export const CODE_TO_PLANE_CODES = 120

const CODE_TO_PLANE = new Uint8Array([
  0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1A,
  0x26, 0x2A, 0x38, 0x05, 0x37, 0x39, 0x15, 0x1B, 0x36, 0x3A,
  0x25, 0x2B, 0x48, 0x04, 0x47, 0x49, 0x14, 0x1C, 0x35, 0x3B,
  0x46, 0x4A, 0x24, 0x2C, 0x58, 0x45, 0x4B, 0x34, 0x3C, 0x03,
  0x57, 0x59, 0x13, 0x1D, 0x56, 0x5A, 0x23, 0x2D, 0x44, 0x4C,
  0x55, 0x5B, 0x33, 0x3D, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1E,
  0x66, 0x6A, 0x22, 0x2E, 0x54, 0x5C, 0x43, 0x4D, 0x65, 0x6B,
  0x32, 0x3E, 0x78, 0x01, 0x77, 0x79, 0x53, 0x5D, 0x11, 0x1F,
  0x64, 0x6C, 0x42, 0x4E, 0x76, 0x7A, 0x21, 0x2F, 0x75, 0x7B,
  0x31, 0x3F, 0x63, 0x6D, 0x52, 0x5E, 0x00, 0x74, 0x7C, 0x41,
  0x4F, 0x10, 0x20, 0x62, 0x6E, 0x30, 0x73, 0x7D, 0x51, 0x5F,
  0x40, 0x72, 0x7E, 0x61, 0x6F, 0x50, 0x71, 0x7F, 0x60, 0x70,
])

/**
 * Reverse lookup: raw distance value `r` ∈ [1..120] ↔ packed (y, x). We
 * build this once on module load — 120 entries, takes microseconds.
 *
 * Each entry stores `(yoffset << 4) | (xoffset & 0x0F)` from the spec
 * table; the encoder uses these to score candidate plane codes for a
 * given linear distance.
 */
const PLANE_TO_RAW: Map<number, number> = (() => {
  const m = new Map<number, number>()
  for (let i = 0; i < CODE_TO_PLANE_CODES; i++) {
    m.set(CODE_TO_PLANE[i], i + 1)
  }
  return m
})()

/**
 * Convert a raw distance value (1..MAX_DISTANCE) into the actual linear
 * pixel offset, using the plane-code mapping for raw values ≤ 120. The
 * `xsize` argument is the width of the pixel stream the backreference
 * lives in — the same width passed to `decodePixelStream`.
 */
export function planeCodeToDistance(xsize: number, rawDistance: number): number {
  if (rawDistance > CODE_TO_PLANE_CODES) {
    return rawDistance - CODE_TO_PLANE_CODES
  }
  const planeCode = CODE_TO_PLANE[rawDistance - 1]
  const yoffset = planeCode >>> 4
  const xoffset = 8 - (planeCode & 0x0F)
  const dist = yoffset * xsize + xoffset
  return dist >= 1 ? dist : 1
}

/**
 * Convert a linear pixel distance into the *cheapest* raw distance value
 * — accounting for the plane-code shortcuts. For the same final distance
 * there can be (a) the linear-form `distance + 120`, and (b) zero or
 * more plane-code forms within ±7 columns of the current pixel. We
 * return the raw value with the shortest prefix code; ties broken by
 * smaller raw value (matches libwebp's preference order).
 */
export function linearDistanceToRaw(distance: number, xsize: number): number | null {
  if (distance < 1) return null

  let bestRaw = distance + CODE_TO_PLANE_CODES
  if (bestRaw > MAX_DISTANCE) bestRaw = -1
  let bestCost = bestRaw > 0 ? rawCost(bestRaw) : Infinity

  // Plane-code candidates: yoffset 0..7, xoffset -7..+8, where
  //   distance = yoffset * xsize + xoffset
  // yoffset is bounded by 7 (4-bit, but only 0..7 appear in the table —
  // top bit is 0 in every entry).
  for (let y = 0; y <= 7; y++) {
    const x = distance - y * xsize
    if (x < -7 || x > 8 || x === 0 && y === 0) continue
    const planeCode = ((y & 0x07) << 4) | ((8 - x) & 0x0F)
    const raw = PLANE_TO_RAW.get(planeCode)
    if (raw === undefined) continue
    const cost = rawCost(raw)
    if (cost < bestCost || (cost === bestCost && raw < bestRaw)) {
      bestRaw = raw
      bestCost = cost
    }
  }

  return bestRaw > 0 ? bestRaw : null
}

/** Cost (in extra bits beyond the prefix code) of a raw distance. */
function rawCost(raw: number): number {
  if (raw <= 4) return 0
  for (let code = 4; code < NUM_DISTANCE_CODES; code++) {
    const base = DISTANCE_OFFSET[code]
    const extraBits = DISTANCE_EXTRA_BITS[code]
    if (raw >= base && raw < base + (1 << extraBits)) return extraBits
  }
  return Infinity
}
