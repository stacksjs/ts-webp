/**
 * VP8L distance-code helpers.
 *
 * Distances in VP8L are LZ77 backreferences expressed as a single prefix
 * code 0..39 plus extra bits for codes ≥ 4. The code → distance mapping
 * is exactly DEFLATE-style: each code represents a base distance with a
 * fixed number of extra bits to read.
 *
 * The full VP8L spec also defines a 120-entry "near distance" table
 * (`kCodeToPlane` in libwebp) that maps small raw distances to 2D
 * (xi, yi) offsets, so a backreference like "the pixel directly above
 * me" can be expressed as a 1-2 bit code instead of an `xsize`-bit code.
 * We *don't* implement that here: it bakes `xsize` into the bitstream in
 * a way that's brittle to round-trip and only matters for libwebp
 * interop, which we already give up on (we don't support libwebp's
 * pre-transforms either). Encoder and decoder treat distance codes as
 * pure linear pixel offsets.
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
 * already read, return the linear pixel distance.
 */
export function distanceFromCode(code: number, extraBitsValue: number): number {
  if (code < 0 || code >= NUM_DISTANCE_CODES) {
    throw new Error(`distanceFromCode: code ${code} out of range`)
  }
  if (code < 4) return code + 1
  return DISTANCE_OFFSET[code] + extraBitsValue
}

/**
 * For encoding: convert a linear pixel distance into the prefix code +
 * extra-bits payload. Returns `null` if the distance is out of range
 * (≤ 0 or > `MAX_DISTANCE`).
 */
export function distanceToCode(distance: number): {
  code: number
  extraBits: number
  extraValue: number
} | null {
  if (distance < 1 || distance > MAX_DISTANCE) return null
  if (distance <= 4) return { code: distance - 1, extraBits: 0, extraValue: 0 }
  // Binary search would be marginally faster, but the alphabet is 40
  // entries — linear scan is fine and easier to audit.
  for (let code = 4; code < NUM_DISTANCE_CODES; code++) {
    const base = DISTANCE_OFFSET[code]
    const extraBits = DISTANCE_EXTRA_BITS[code]
    if (distance >= base && distance < base + (1 << extraBits)) {
      return { code, extraBits, extraValue: distance - base }
    }
  }
  return null
}
