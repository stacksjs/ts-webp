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

- ✅ **Lossless encode** (VP8L) — produces spec-compliant bitstreams with a
  proper canonical Huffman codec
- ✅ **Lossless decode** (VP8L) — reads our own output and other VP8L
  bitstreams that don't use pre-transforms
- ✅ **Exact round-trip** — `decode(encode(image)).data === image.data`
- ✅ **Alpha channel** — full RGBA support
- ✅ **RIFF container** — parse + emit
- 🚧 **VP8L pre-transforms** (subtract-green, predictor, color,
  color-indexing) — *not yet* supported on either side. The encoder emits
  no transforms (so output is correct but not as small as libwebp's); the
  decoder rejects any bitstream that uses them rather than mangling
  output silently. Most photo-quality libwebp output uses subtract-green
  by default, so don't expect to decode arbitrary `.webp` files yet.
- 🚧 **LZ77 backreferences** + **color cache** — encoder doesn't emit
  these yet, so compression on photos is around 70-80 % of raw RGBA.
  Decoder handles both per spec.
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

Reference numbers for a 256×256 image (262 KB raw RGBA) with mixed pixel
content, on Apple M-series, Bun 1.3:

| | Time |
|---|---|
| Encode | ~3.7 ms |
| Decode | ~1.4 ms |
| Output size | 78 % of raw |

The hot paths use 32-bit accumulator-based bit I/O, `Uint32Array`-backed
ARGB pixel buffers, and a primary-LUT-first Huffman decoder. Adding
subtract-green + LZ77 + color cache (planned) drops the output size to
~30-50 % of raw for natural images.

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
