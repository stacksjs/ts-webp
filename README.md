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
- ✅ **Animated WebP** (ANIM / ANMF) — full container support; per-frame
  payload routed to the VP8L (lossless), VP8 (lossy), or VP8 + ALPH
  (lossy + alpha plane) decoder based on the inner chunks present
- ✅ **Extended format** (VP8X) — produced by `encodeWithAlpha`,
  consumed transparently by `decode`
- ✅ **Lossy + alpha** (VP8X + ALPH + VP8) — full ALPH chunk support
  including filter inversion (none / horizontal / vertical / gradient)
  and lossless alpha plane (method=0 raw bytes, method=1 VP8L green
  channel). `alpha-uncompressed` fixtures match dwebp bit-exactly;
  `alpha-q80` matches RGB bit-exactly with bounded alpha drift on the
  cwebp predictor + cache path (see test/alpha.test.ts for details)

### Lossy (VP8)
- ✅ **Boolean arithmetic decoder** — full 32-bit-register implementation
  per RFC 6386 §7
- ✅ **Frame header parsing** — keyframe detection, segmentation, loop-
  filter parameters, quantiser indices, partition layout, coef-prob
  update loop, mb_no_skip_coef
- ✅ **Mode-info decoding** — keyframe Y mode (DC/V/H/TM/B_PRED), 16
  per-subblock B-modes with contextual neighbour probabilities, UV mode
- ✅ **Coefficient decoding** — DCT token tree walk via boolean coder,
  CAT1..CAT6 magnitude tokens, sign bits, zigzag scan
- ✅ **Inverse transforms** — 4×4 IDCT and 4×4 WHT (Y2) bit-exact with
  libvpx
- ✅ **Intra prediction** — DC/V/H/TM (16×16 luma, 8×8 chroma) and all 10
  B-modes (4×4 luma) per RFC 6386 §12
- ✅ **Loop filter** — simple + normal filters with HEV-conditional
  kernel, MB-edge and sub-block-edge variants, per-level threshold
  derivation per RFC 6386 §15
- ✅ **YUV 4:2:0 → RGBA** — BT.601 fixed-point conversion
- ✅ **Bit-exact with `dwebp`** — the decoder is a top-to-bottom port
  of libwebp's reference implementation (probabilities, quantiser
  tables, bool decoder, coefficient decoder, IDCT, intra prediction,
  loop filter, fancy chroma upsampling, YUV→RGB conversion) and
  matches `dwebp` exactly across the full quality range. The test
  suite verifies bit-exact output (max-pixel-diff = 0) on:
    - 16×16 single-MB B_PRED at q=75
    - 32×16 multi-MB gradient at q=75
    - 32×16 solid colour at q=75
    - 256×192 multi-segment gradient at q=30, q=50, q=75, q=90
    - 384×288 multi-partition photo-like at q=30, q=75, q=95
- ❌ **Encode** — not implemented. VP8 lossy encoding requires forward
  DCT/WHT, a boolean arithmetic encoder, intra-mode selection over the
  full 4×4 + 16×16 mode set, residual token coding, segment/quantiser
  rate control, and bit-exact tables (probabilities, dequant LUTs,
  zigzag scan, coefficient bands, Q-index → DC/AC quantiser maps).
  This is a multi-thousand-line undertaking; out of scope for this
  package. Use `cwebp` to produce lossy WebP files.

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
├── decoder.ts            — RIFF dispatch → VP8L / VP8 / VP8 + ALPH
├── encoder.ts            — entry point; chooses lossless or lossy
├── riff.ts               — RIFF container parse/build + WebP info parsing
├── bitreader.ts          — BitReader / BitWriter (32-bit accumulator)
├── animation.ts          — ANIM/ANMF container parser; per-frame routing
├── alpha.ts              — ALPH chunk decoder (raw / VP8L green) + filter inverse
├── vp8/
│   ├── decoder.ts        — VP8 (lossy) decoder, bit-exact with libwebp
│   ├── bool-decoder.ts   — arithmetic boolean decoder
│   ├── header.ts         — frame header parser
│   ├── intra.ts          — intra prediction modes (4×4 + 16×16 + UV)
│   ├── coeff.ts          — DCT coefficient tree decoder
│   ├── idct.ts           — 4×4 IDCT + Y2 WHT inverse
│   ├── loop-filter.ts    — simple + normal in-loop filters
│   ├── tables.ts         — quantiser / probability / dequant tables
│   └── mode.ts           — mode-info decoder
└── vp8l/
    ├── encoder.ts        — VP8L (lossless) encoder
    ├── decoder.ts        — VP8L (lossless) decoder; also exports
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
