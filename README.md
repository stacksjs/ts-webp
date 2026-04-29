# ts-webp

A pure-TypeScript WebP encoder/decoder with zero runtime dependencies.

```ts
import { decode, encode } from 'ts-webp'

// Encode RGBA pixels to a lossless WebP buffer.
const encoded = encode({ data: rgba, width, height })

// Decode back to RGBA pixels — exact byte-for-byte round-trip.
const { data, width, height, hasAlpha } = decode(encoded)
```

## Status

### Lossless (VP8L) — fully working
- ✅ **Encode** with subtract-green, predictor, color, color-indexing
  transforms, LZ77 backreferences, and color cache
- ✅ **Decode** with all four pre-transforms, color cache, and the
  meta-Huffman image (multi-group bitstreams, used by libwebp at
  quality > 75)
- ✅ **Exact round-trip** — `decode(encode(image)).data === image.data`
  byte-for-byte across every test image
- ✅ **Alpha channel** — full RGBA support

### Container / animation
- ✅ **RIFF / WEBP** — parse + emit, zero-copy chunk extraction
- ✅ **Animated WebP** (ANIM / ANMF) — full container support; each
  frame decoded via the lossless path. Lossy frames fall through to the
  VP8 decoder (currently unsupported, see below).
- ✅ **Extended format** (VP8X) — produced by `encodeWithAlpha`,
  consumed transparently by `decode`.

### Lossy (VP8)
- ✅ **Boolean arithmetic decoder** — full implementation per RFC 6386
- ✅ **Frame header parsing** — keyframe detection, segmentation, loop-
  filter parameters, quantiser indices, partition layout. Useful on its
  own for introspection (`getWebpInfo`) and as a foundation for the
  pixel decoder.
- 🚧 **Pixel decode** — coefficient decoding, inverse DCT/WHT, intra
  prediction (4×4 and 16×16 luma + chroma modes), loop filter, and
  YUV→RGB conversion still need to be wired up. `decodeVP8` parses
  the header, then throws with a precise "pixel decode not yet
  implemented" message. A complete VP8 decoder is ~1500-2000 lines and
  realistically needs validation against a reference corpus to ship
  responsibly.
- ❌ **Encode** — not implemented. VP8 lossy encoding additionally
  needs DCT/quantisation, intra-mode selection, rate control, and the
  boolean encoder. Out of scope; use `cwebp` for lossy WebP files.

## Install

```bash
bun add ts-webp
```

## API

### `encode(image, options?) → Uint8Array`

```ts
encode({ data: Uint8Array, width: number, height: number, hasAlpha?: boolean }, {
  lossless?: boolean // default: true; false silently falls back to lossless
})
```

Returns a complete `RIFF/WEBP/VP8L` byte stream.

### `decode(buffer, options?) → { data, width, height, hasAlpha }`

```ts
decode(Uint8Array | ArrayBuffer, {
  format?: 'rgba' | 'rgb' // default: 'rgba'
})
```

Returns RGBA (or RGB) pixel data.

### `encodeWithAlpha(image, options?) → Uint8Array`

Variant that builds a `VP8X + ALPH + VP8L` extended-format container. Use
this when you need to carry alpha through a pipeline that depends on the
extended-format flag — `encode` already preserves alpha through the
single-chunk VP8L route.

### Lower-level helpers

- `parseRiff(buffer) → RiffChunk[]` — RIFF container reader
- `createRiffContainer(chunks) → Uint8Array` — RIFF container builder
- `getWebpInfo(chunks) → WebpInfo` — width/height/flags from chunks

## Performance

Reference numbers on Apple M-series, Bun 1.3, 256×256 photo-like image:

| | Time |
|---|---|
| Encode | ~2.2 ms |
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

## Encoder options

```ts
encode(image, {
  lossless?: boolean        // default: true; false silently falls back to lossless
  // Internal feature toggles (mainly for tests / debugging):
  subtractGreen?: boolean   // default: true
  useLZ77?: boolean         // default: true
  useColorCache?: boolean   // default: true
  cacheBits?: number        // default: 11; range 1..11
})
```

Each transform is independently toggleable: `{ subtractGreen: false }`
for example produces a valid VP8L bitstream that just skips the
subtract-green pass. We use this in tests to isolate per-feature
round-trip correctness.

## Architecture

```
src/
├── index.ts              — public API
├── decoder.ts            — RIFF dispatch → VP8L / VP8 decoders
├── encoder.ts            — entry point; chooses lossless or lossy
├── riff.ts               — RIFF container parse/build + WebP info parsing
├── bitreader.ts          — BitReader / BitWriter (32-bit accumulator)
├── vp8/
│   └── decoder.ts        — VP8 (lossy) — currently throws
└── vp8l/
    ├── encoder.ts        — VP8L (lossless) encoder
    ├── decoder.ts        — VP8L (lossless) decoder
    └── huffman.ts        — canonical Huffman codec (encode + decode)
```

## License

MIT
