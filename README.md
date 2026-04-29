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

- ✅ **Lossless encode** (VP8L) — spec-compliant bitstream with a
  canonical Huffman codec, subtract-green pre-transform, LZ77
  backreferences, and color cache
- ✅ **Lossless decode** (VP8L) — reads our own output and any other
  VP8L bitstream that uses only subtract-green among the pre-transforms
- ✅ **Exact round-trip** — `decode(encode(image)).data === image.data`
  for every image we know how to construct, byte-for-byte
- ✅ **Alpha channel** — full RGBA support
- ✅ **RIFF container** — parse + emit
- 🚧 **Predictor / color / color-indexing transforms** — encoder doesn't
  emit them; decoder rejects any bitstream that does (so we don't
  silently mis-decode libwebp output that uses them). The `subtract-green`
  transform — which is the *most common* one in libwebp output — is
  fully supported.
- ❌ **Lossy** (VP8) encoding — not implemented; `encode(…, { lossless:
  false })` falls back to lossless. VP8 is a multi-month port of libvpx
  and out of scope for this library.
- ❌ **Lossy** (VP8) decoding — not implemented; `decode` throws a clear
  error on any `VP8 ` chunk rather than producing fake gray output.
- ❌ **Animation** (ANIM/ANMF) — container parses but frames don't decode.

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
