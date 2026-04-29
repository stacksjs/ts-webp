/**
 * VP8L color-indexing transform (transform type 3).
 *
 * When an image has at most 256 distinct ARGB values — typical of icons,
 * UI assets, screenshots, and most non-photo content — replace each
 * pixel with its index into a palette. The 256-entry palette is sent
 * once at the start of the bitstream as a 1-row sub-image; the main
 * pixel stream then carries indices in the green channel only.
 *
 * For palettes of size ≤ 16 there's an extra trick: pack multiple
 * indices into a single output pixel, shrinking the encoded image's
 * width by the corresponding factor. A 4-color palette packs 4 pixels
 * into one byte (2 bits each); the encoded width becomes
 * `ceil(width / 4)`. The decoder un-packs after pixel decode.
 *
 * Reference: WebP Lossless Bitstream Specification §3.4.
 */

/** Maximum palette size the spec allows (8-bit `color_table_size - 1`). */
export const MAX_PALETTE_SIZE = 256

/**
 * Packing factor for palette sizes ≤ 16: how many pixel indices share
 * a single byte. The spec is fixed; this is a lookup-table summary.
 */
export function packingFactor(paletteSize: number): number {
  if (paletteSize <= 2) return 8
  if (paletteSize <= 4) return 4
  if (paletteSize <= 16) return 2
  return 1
}

/**
 * Try to apply the forward color-indexing transform to an ARGB image.
 *
 * Returns `null` if the image has more than 256 distinct colors (the
 * transform isn't applicable). Otherwise mutates `argb` in place: each
 * pixel becomes `(0xFF << 24) | (paletteIndex << 8)`. For small palettes,
 * the caller also gets back an "encoded width" smaller than the input —
 * pixels in `argb[0..encodedWidth*height]` then hold up-to-`packingFactor`
 * indices each, packed in the green channel low bits.
 *
 * Layout of palette indices in a packed pixel's green channel:
 *   bit 0..(bpp-1)      = first sub-pixel (leftmost in unpacked coords)
 *   bit bpp..(2bpp-1)   = second sub-pixel
 *   ...
 * where `bpp = 8 / packingFactor`.
 *
 * The palette is delta-encoded by the caller (the bitstream uses
 * `palette[i] - palette[i-1]` so the wire-form palette compresses
 * better; we leave that delta encoding to the bitstream emitter).
 */
export function applyColorIndexTransform(
  argb: Uint32Array,
  width: number,
  height: number,
): { palette: Uint32Array, encodedWidth: number, encodedArgb: Uint32Array } | null {
  // Build palette via a single-pass scan. We use a Map<number, number>
  // keyed by ARGB-as-number; for a 256-color cap that's a few hundred
  // entries and Map lookups are fine at that scale.
  const paletteMap = new Map<number, number>()
  for (let i = 0; i < argb.length; i++) {
    const px = argb[i]
    if (!paletteMap.has(px)) {
      if (paletteMap.size >= MAX_PALETTE_SIZE) return null
      paletteMap.set(px, paletteMap.size)
    }
  }

  const paletteSize = paletteMap.size
  const palette = new Uint32Array(paletteSize)
  for (const [px, idx] of paletteMap) palette[idx] = px

  const factor = packingFactor(paletteSize)
  const encodedWidth = (width + factor - 1) >>> Math.log2(factor)
  const bitsPerIndex = 8 / factor

  // Encoded image: encodedWidth × height pixels, each holding up to
  // `factor` palette indices in its green channel.
  const encodedArgb = new Uint32Array(encodedWidth * height)
  for (let y = 0; y < height; y++) {
    for (let xe = 0; xe < encodedWidth; xe++) {
      let packed = 0
      for (let k = 0; k < factor; k++) {
        const xs = xe * factor + k
        if (xs >= width) break
        const idx = paletteMap.get(argb[y * width + xs])!
        packed |= idx << (k * bitsPerIndex)
      }
      // Always emit opaque-alpha encoded pixels; index lives in green.
      encodedArgb[y * encodedWidth + xe] = (0xFF << 24) | (packed << 8)
    }
  }

  return { palette, encodedWidth, encodedArgb }
}

/**
 * Apply the inverse color-indexing transform: take the decoded encoded
 * image (whose green channel carries packed indices) and the palette,
 * and produce a fresh ARGB array of original-width pixels.
 *
 * The caller passes in the post-delta-decode palette (i.e. already-
 * accumulated absolute palette values, not the wire-form deltas).
 */
export function inverseColorIndexTransform(
  encodedArgb: Uint32Array,
  encodedWidth: number,
  width: number,
  height: number,
  palette: Uint32Array,
): Uint32Array {
  const factor = packingFactor(palette.length)
  const bitsPerIndex = 8 / factor
  const indexMask = (1 << bitsPerIndex) - 1
  const out = new Uint32Array(width * height)

  for (let y = 0; y < height; y++) {
    for (let xe = 0; xe < encodedWidth; xe++) {
      const packed = (encodedArgb[y * encodedWidth + xe] >>> 8) & 0xFF
      for (let k = 0; k < factor; k++) {
        const xs = xe * factor + k
        if (xs >= width) break
        const idx = (packed >>> (k * bitsPerIndex)) & indexMask
        // Defensive bound check — a corrupted bitstream could have an
        // index past the palette, in which case we'd otherwise read
        // `undefined`. Fall through to opaque black for malformed inputs.
        out[y * width + xs] = idx < palette.length ? palette[idx] : 0xFF000000 | 0
      }
    }
  }
  return out
}

/**
 * Delta-encode a palette for the wire. The bitstream stores
 * `palette[0]` directly and then each `palette[i] - palette[i-1]`
 * (channel-wise mod 256). On a smoothly-varying palette (typical of
 * antialiased UI / iconography) this leaves the differences with
 * many zero or near-zero channels — much cheaper Huffman codes.
 */
export function deltaEncodePalette(palette: Uint32Array): Uint32Array {
  const out = new Uint32Array(palette.length)
  out[0] = palette[0]
  for (let i = 1; i < palette.length; i++) {
    out[i] = subArgbBytewise(palette[i], palette[i - 1])
  }
  return out
}

/** Reverse of `deltaEncodePalette`. */
export function deltaDecodePalette(deltaPalette: Uint32Array): Uint32Array {
  const out = new Uint32Array(deltaPalette.length)
  out[0] = deltaPalette[0]
  for (let i = 1; i < deltaPalette.length; i++) {
    out[i] = addArgbBytewise(deltaPalette[i], out[i - 1])
  }
  return out
}

function subArgbBytewise(a: number, b: number): number {
  return (
    (((a >>> 24) - (b >>> 24)) & 0xFF) << 24
    | (((a >>> 16) - (b >>> 16)) & 0xFF) << 16
    | (((a >>> 8) - (b >>> 8)) & 0xFF) << 8
    | ((a - b) & 0xFF)
  ) >>> 0
}

function addArgbBytewise(a: number, b: number): number {
  return (
    (((a >>> 24) + (b >>> 24)) & 0xFF) << 24
    | (((a >>> 16) + (b >>> 16)) & 0xFF) << 16
    | (((a >>> 8) + (b >>> 8)) & 0xFF) << 8
    | ((a + b) & 0xFF)
  ) >>> 0
}
