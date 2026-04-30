# ts-webp

A pure-TypeScript WebP encoder/decoder with zero runtime dependencies.

```ts
import { decode, encode } from 'ts-webp'

// Encode RGBA pixels to a WebP buffer (lossless by default).
const encoded = encode({ data: rgba, width, height })

// Decode back to RGBA pixels.
const { data, width, height, hasAlpha } = decode(encoded)
```

## What's included

A complete WebP codec: lossless (VP8L) and lossy (VP8) encode + decode,
the RIFF / VP8X container, the ALPH alpha-plane chunk for lossy + alpha
files, and animated WebP playback (ANMF frames routed to whichever
inner codec they carry).

The lossless path is bit-exact — `decode(encode(image))` returns the
input bytes unchanged across every test image. The lossy path is
spec-compliant in both directions: our decoder matches `dwebp` to the
last bit on a representative fixture set, and our encoder produces
streams that `dwebp` decodes cleanly.

## Install

```bash
bun add ts-webp
```

## API

### `encode(image, options?) → Uint8Array`

Encode an RGBA image to a complete `RIFF/WEBP` byte stream.

```ts
encode(
  { data: Uint8Array, width: number, height: number, hasAlpha?: boolean },
  {
    lossless?: boolean   // default true → VP8L; false → VP8
    quality?: number     // 0..100, lossy only (default 75)
    // Lossless-only feature toggles (for tests / debugging):
    subtractGreen?: boolean   // default true
    useLZ77?: boolean         // default true
    useColorCache?: boolean   // default true
    cacheBits?: number        // default 11; range 1..11
  },
)
```

Lossless output (`VP8L` chunk) is bit-exact through `decode`. Lossy
output (`VP8 ` chunk) goes through forward DCT + quantisation; quality
trades size for fidelity monotonically — a higher `quality` produces a
larger, less-distorted file.

### `decode(buffer, options?) → { data, width, height, hasAlpha }`

```ts
decode(
  Uint8Array | ArrayBuffer,
  { format?: 'rgba' | 'rgb' },  // default 'rgba'
)
```

Auto-detects the inner codec — VP8L, VP8, VP8 + ALPH, or VP8X-wrapped
variants — and returns RGBA (or RGB) pixel data.

### `encodeWithAlpha(image, options?) → Uint8Array`

Builds a `VP8X + VP8L` extended-format container. Use this when you
need the VP8X header flag set (e.g. for tooling that branches on it);
`encode` already carries alpha through the simpler single-chunk VP8L
route, so for most callers `encode` is enough.

### `decodeAnimation(buffer) → { width, height, loopCount, backgroundColor, frames[] }`

Parses an animated WebP and returns each frame's pixels (RGBA), offset
within the canvas, duration, blend mode, and dispose method. Frames
are routed to the lossless, lossy, or lossy + alpha decoder based on
the inner chunks each ANMF block carries.

### Lower-level helpers

- `parseRiff(buffer) → RiffChunk[]` — RIFF container reader
- `createRiffContainer(chunks) → Uint8Array` — RIFF container builder
- `getWebpInfo(chunks) → WebpInfo` — width/height/flags from chunks

## Codec coverage

**Lossless (VP8L)** — full pre-transform stack (subtract-green,
predictor, color, color-indexing), LZ77 with the spec's 120-entry
plane-code distance map, and a color cache. Decode also handles the
meta-Huffman image cwebp emits at quality > 75. Both directions are
bit-exact across all test images, with single-symbol Huffman trees
collapsing to 0 bits per symbol the way libwebp's `BuildHuffmanTable`
does.

**Lossy (VP8)** — keyframe decode is a top-to-bottom port of libwebp's
reference implementation (boolean coder, frame header, segmentation,
filter header, quantiser tables, coefficient token tree, 4×4 IDCT, Y2
WHT, all 14 predictor modes, simple + normal loop filter, fancy
chroma upsampling, BT.601 YUV→RGB) and matches `dwebp` byte-for-byte
across fixtures from 16×16 single-MB up through 384×288 multi-partition
photo-like content.

The encode side is a smaller and more pragmatic implementation: 16×16
intra-DC prediction for every macroblock, single segment, single token
partition, no loop filter, default coefficient probabilities. Output
is decode-correct (bytes parse, pixels match within the quantiser
budget) but not rate-distortion-competitive with `cwebp` — there's no
mode search and no rate control. Use it for portability or for cases
where you control the producer; reach for `cwebp` when you need the
smallest possible output for a given quality target.

**Container** — the RIFF / VP8X parser/builder handles every WebP
flavour we encode or decode: simple `VP8L` or `VP8 ` files, `VP8X` +
`VP8L` for extended-format lossless, `VP8X + ALPH + VP8 ` for lossy
images that carry an alpha plane (with both raw-byte and lossless-VP8L
green-channel encoding methods), and animated WebP via `ANIM` / `ANMF`.

The ALPH-chunk decoder applies the four filter inverses (none,
horizontal, vertical, gradient) per RFC 9649 §9. Animated WebP frames
are routed to the appropriate inner decoder per-frame, so a single
animation can mix lossless and lossy frames.

## Performance

Reference numbers on Apple M-series, Bun 1.3, 256×256 photo-like image:

| | Time |
|---|---|
| Encode (lossless) | ~2.2 ms |
| Decode | ~2.5 ms |
| Output size | 15 % of raw RGBA |

Compression ratios across patterns (with all features on):

| pattern (64×64) | output / raw |
|---|---|
| single colour | 8 % |
| stripes (2 colours) | 3 % |
| 8-colour palette, scattered | 3 % |
| RGB-correlated channels | 16 % |
| photo-like sin/cos blend | 18 % |

The hot paths use 32-bit accumulator-based bit I/O, `Uint32Array`-backed
ARGB pixel buffers, a primary-LUT-first Huffman decoder, and a hash-table
LZ77 match finder over 3-pixel ARGB windows.

## Architecture

```
src/
├── index.ts              — public API
├── decoder.ts            — RIFF dispatch → VP8L / VP8 / VP8 + ALPH
├── encoder.ts            — entry point; chooses lossless or lossy
├── riff.ts               — RIFF container parse/build + WebP info parsing
├── bitreader.ts          — BitReader / BitWriter (32-bit accumulator)
├── animation.ts          — ANIM/ANMF container parser; per-frame routing
├── alpha.ts              — ALPH chunk decoder + filter inverse
├── vp8/
│   ├── decoder.ts        — VP8 keyframe decoder (libwebp-bit-exact)
│   ├── encoder.ts        — VP8 keyframe encoder (16×16 intra-DC)
│   ├── bool-decoder.ts   — arithmetic boolean decoder
│   ├── bool-encoder.ts   — arithmetic boolean encoder
│   ├── header.ts         — frame header parser
│   ├── intra.ts          — intra prediction modes (4×4 + 16×16 + UV)
│   ├── coeff.ts          — DCT coefficient tree decoder
│   ├── idct.ts           — 4×4 IDCT + Y2 WHT inverse
│   ├── fdct.ts           — 4×4 forward DCT + Y2 forward WHT
│   ├── quant-enc.ts      — round-to-nearest quantiser
│   ├── loop-filter.ts    — simple + normal in-loop filters
│   ├── tables.ts         — quantiser / probability / dequant tables
│   └── mode.ts           — mode-info decoder
└── vp8l/
    ├── encoder.ts        — VP8L lossless encoder
    ├── decoder.ts        — VP8L lossless decoder; also exports
    │                       `decodeVP8LImageStream` for ALPH lossless
    ├── huffman.ts        — canonical Huffman codec (encode + decode)
    ├── distance.ts       — LZ77 distance + plane-code remap
    ├── length.ts         — LZ77 length codes
    ├── predictor.ts      — predictor transform (forward + inverse)
    ├── color.ts          — color transform (forward + inverse)
    └── color-index.ts    — color-indexing transform (forward + inverse)
```

## License

MIT
