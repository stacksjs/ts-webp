/**
 * VP8L length-code helpers — symmetric to `distance.ts`, but for the
 * length component of an LZ77 backreference.
 *
 * Length codes occupy positions 256..279 in the green/length alphabet
 * (i.e. the green tree dispatches to a length code by emitting a symbol
 * ≥ 256). Each code resolves to a base length plus an extra-bits payload
 * using VP8L's own prefix scheme (spec §4.2.1.1, libwebp `PrefixDecode`)
 * — NOT deflate's tables:
 *
 *   code < 4:  value = code + 1, no extra bits
 *   code ≥ 4:  extra  = (code - 2) >> 1
 *              offset = (2 + (code & 1)) << extra
 *              value  = offset + extraBits + 1
 *
 * The same scheme drives the distance alphabet in `distance.ts`.
 */

/** Length codes in the alphabet (positions 256..279 of the green tree). */
export const NUM_LENGTH_CODES = 24

/** Extra-bit count per length code: (code - 2) >> 1 for codes ≥ 4. */
export const LENGTH_EXTRA_BITS = new Uint8Array([
  0, 0, 0, 0, 1, 1, 2, 2,
  3, 3, 4, 4, 5, 5, 6, 6,
  7, 7, 8, 8, 9, 9, 10, 10,
])

/** Base length for each code; final length = base + extraBitsValue. */
export const LENGTH_OFFSET = new Uint16Array([
  1, 2, 3, 4, 5, 7, 9, 13,
  17, 25, 33, 49, 65, 97, 129, 193,
  257, 385, 513, 769, 1025, 1537, 2049, 3073,
])

/** Maximum length representable in a single backreference. */
export const MAX_LENGTH = LENGTH_OFFSET[NUM_LENGTH_CODES - 1]
  + (1 << LENGTH_EXTRA_BITS[NUM_LENGTH_CODES - 1]) - 1 // 3073 + 1023 = 4096

/**
 * For decoding: given a length prefix code 0..23 and the value of its
 * extra bits, return the actual run length (≥ 1).
 */
export function lengthFromCode(code: number, extraBitsValue: number): number {
  if (code < 0 || code >= NUM_LENGTH_CODES) {
    throw new Error(`lengthFromCode: code ${code} out of range`)
  }
  return LENGTH_OFFSET[code] + extraBitsValue
}

/**
 * For encoding: convert a desired run length into the prefix code +
 * extra-bits payload. Returns `null` if `length < 1` or `> MAX_LENGTH`.
 */
export function lengthToCode(length: number): {
  code: number
  extraBits: number
  extraValue: number
} | null {
  if (length < 1 || length > MAX_LENGTH) return null
  // Linear scan over 24 codes — fast enough.
  for (let code = 0; code < NUM_LENGTH_CODES; code++) {
    const base = LENGTH_OFFSET[code]
    const extraBits = LENGTH_EXTRA_BITS[code]
    if (length >= base && length < base + (1 << extraBits)) {
      return { code, extraBits, extraValue: length - base }
    }
  }
  return null
}
