/**
 * VP8L length-code helpers — symmetric to `distance.ts`, but for the
 * length component of an LZ77 backreference.
 *
 * Length codes occupy positions 256..279 in the green/length alphabet
 * (i.e. the green tree dispatches to a length code by emitting a symbol
 * ≥ 256). Each code resolves to a base length plus an extra-bits payload
 * exactly like DEFLATE.
 */

/** Length codes in the alphabet (positions 256..279 of the green tree). */
export const NUM_LENGTH_CODES = 24

/** Extra-bit count per length code. */
export const LENGTH_EXTRA_BITS = new Uint8Array([
  0, 0, 0, 0, 0, 0, 0, 0,
  1, 1, 1, 1, 2, 2, 2, 2,
  3, 3, 3, 3, 4, 4, 4, 4,
])

/** Base length for each code; final length = base + extraBitsValue. */
export const LENGTH_OFFSET = new Uint16Array([
  1, 2, 3, 4, 5, 6, 7, 8,
  9, 11, 13, 15, 17, 21, 25, 29,
  33, 41, 49, 57, 65, 81, 97, 113,
])

/** Maximum length representable in a single backreference. */
export const MAX_LENGTH = LENGTH_OFFSET[NUM_LENGTH_CODES - 1]
  + (1 << LENGTH_EXTRA_BITS[NUM_LENGTH_CODES - 1]) - 1 // 113 + 15 = 128

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
