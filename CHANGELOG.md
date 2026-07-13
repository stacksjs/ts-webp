# Changelog

[Compare changes](https://github.com/stacksjs/ts-webp/compare/v0.1.0...v0.1.1)

## 🚀 Features

- **encoder**: add cwebp CLI backend with auto fallback ([64621a3](https://github.com/stacksjs/ts-webp/commit/64621a3)) _(by Chris <chrisbreuer93@gmail.com>)_
- **vp8**: port libwebp's decoder verbatim — bit-exact for solid-colour, near-exact for B_PRED ([c179faf](https://github.com/stacksjs/ts-webp/commit/c179faf)) _(by Chris <chrisbreuer93@gmail.com>)_
- **vp8**: full lossy decoder pipeline — coefficient decode, IDCT, intra prediction, loop filter ([61e9924](https://github.com/stacksjs/ts-webp/commit/61e9924)) _(by Chris <chrisbreuer93@gmail.com>)_
- **vp8**: boolean decoder + frame-header parser ([306df93](https://github.com/stacksjs/ts-webp/commit/306df93)) _(by Chris <chrisbreuer93@gmail.com>)_
- animated WebP decode (ANIM/ANMF chunks) ([c451517](https://github.com/stacksjs/ts-webp/commit/c451517)) _(by Chris <chrisbreuer93@gmail.com>)_
- **vp8l**: meta-Huffman image + zero-copy RIFF + format:'rgb' tests ([c467470](https://github.com/stacksjs/ts-webp/commit/c467470)) _(by Chris <chrisbreuer93@gmail.com>)_
- **vp8l**: color + color-indexing transforms — full transform-chain support ([c3507db](https://github.com/stacksjs/ts-webp/commit/c3507db)) _(by Chris <chrisbreuer93@gmail.com>)_

## 🐛 Bug Fixes

- **vp8**: advance coefficient contexts ([dfc4e15](https://github.com/stacksjs/ts-webp/commit/dfc4e15)) _(by Chris <chrisbreuer93@gmail.com>)_
- **vp8**: predict from reconstructed neighbors ([fb3b5a0](https://github.com/stacksjs/ts-webp/commit/fb3b5a0)) _(by Chris <chrisbreuer93@gmail.com>)_
- **vp8l**: correct length prefix table and predictor TR wraparound ([45cd77f](https://github.com/stacksjs/ts-webp/commit/45cd77f)) _(by Chris <chrisbreuer93@gmail.com>)_
- **scripts**: stop double-generating CHANGELOG on release ([66a80ea](https://github.com/stacksjs/ts-webp/commit/66a80ea)) _(by Glenn Michael Torregosa <gtorregosa@gmail.com>)_
- **tsc**: stdin sink type, ArrayBuffer variance, test fixes ([afe1c6e](https://github.com/stacksjs/ts-webp/commit/afe1c6e)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- **vp8**: bit-exact dwebp agreement — fancy upsampling + B-mode/16×16 enum alignment ([ebf4cda](https://github.com/stacksjs/ts-webp/commit/ebf4cda)) _(by Chris <chrisbreuer93@gmail.com>)_
- **vp8**: top-right extension at rows 3/7/11 to match libwebp's stride math ([de1f2da](https://github.com/stacksjs/ts-webp/commit/de1f2da)) _(by Chris <chrisbreuer93@gmail.com>)_

## 🤖 Continuous Integration

- drop redundant setup-bun (pantry installs bun via deps.yaml) ([40b3de3](https://github.com/stacksjs/ts-webp/commit/40b3de3)) _(by glennmichael123 <gtorregosa@gmail.com>)_

## 🧹 Chores

- release v0.1.1 ([725b917](https://github.com/stacksjs/ts-webp/commit/725b917)) _(by Chris <chrisbreuer93@gmail.com>)_
- **deps**: add typescript ([a178708](https://github.com/stacksjs/ts-webp/commit/a178708)) _(by Chris <chrisbreuer93@gmail.com>)_
- **deps**: refresh bun.lock to pick up pickier 0.1.37 ([c800c1f](https://github.com/stacksjs/ts-webp/commit/c800c1f)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- **deps**: refresh bun.lock to pick up pickier 0.1.35 ([193f3c5](https://github.com/stacksjs/ts-webp/commit/193f3c5)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- **deps**: refresh bun.lock to pick up pickier 0.1.33 ([b84bc0a](https://github.com/stacksjs/ts-webp/commit/b84bc0a)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- **deps**: refresh bun.lock to pick up @stacksjs/logsmith 0.2.3 ([7497167](https://github.com/stacksjs/ts-webp/commit/7497167)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- **deps**: refresh bun.lock to pick up buddy-bot 0.9.20 ([ff911d2](https://github.com/stacksjs/ts-webp/commit/ff911d2)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- **deps**: bump better-dx to ^0.2.15 ([e17ea8e](https://github.com/stacksjs/ts-webp/commit/e17ea8e)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- **ci**: bump actions/checkout to v6, actions/cache to v5 ([cd43dae](https://github.com/stacksjs/ts-webp/commit/cd43dae)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- refresh bun.lock to pick up bun-plugin-dtsx@0.9.18 ([50b000f](https://github.com/stacksjs/ts-webp/commit/50b000f)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- refresh bun.lock and apply pickier --fix ([ba4ec6e](https://github.com/stacksjs/ts-webp/commit/ba4ec6e)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- refresh bun.lock ([e367c72](https://github.com/stacksjs/ts-webp/commit/e367c72)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- lint:fix ([8bbcef9](https://github.com/stacksjs/ts-webp/commit/8bbcef9)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- refresh bun.lock to pick up latest pickier ([6056514](https://github.com/stacksjs/ts-webp/commit/6056514)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- wip ([34a4145](https://github.com/stacksjs/ts-webp/commit/34a4145)) _(by Chris <chrisbreuer93@gmail.com>)_
- wip ([2e195f2](https://github.com/stacksjs/ts-webp/commit/2e195f2)) _(by Chris <chrisbreuer93@gmail.com>)_

## ⏪ Reverts

- keep staged-lint kebab + bunx gitlint shorthand ([f70c7c0](https://github.com/stacksjs/ts-webp/commit/f70c7c0)) _(by glennmichael123 <gtorregosa@gmail.com>)_

## Contributors

- _Chris <chrisbreuer93@gmail.com>_
- _Glenn Michael Torregosa <gtorregosa@gmail.com>_
- _glennmichael123 <gtorregosa@gmail.com>_

### 🚀 Features

- **vp8l**: predictor transform — 2× compression on photos ([e604db4](https://github.com/stacksjs/ts-webp/commit/e604db4)) _(by Chris <chrisbreuer93@gmail.com>)_
- **vp8l**: subtract-green + LZ77 + color cache → 5× compression on photos ([4c694bd](https://github.com/stacksjs/ts-webp/commit/4c694bd)) _(by Chris <chrisbreuer93@gmail.com>)_

### 🐛 Bug Fixes

- drop inline comment tripping pickier no-unused-vars false positive ([7448506](https://github.com/stacksjs/ts-webp/commit/7448506)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- resolve CI lint and fuzz-test timeout ([f5db6ef](https://github.com/stacksjs/ts-webp/commit/f5db6ef)) _(by glennmichael123 <gtorregosa@gmail.com>)_

### ⚡ Performance Improvements

- real binary heap, parallel-array tokens, two-level Huffman LUT ([52c1df6](https://github.com/stacksjs/ts-webp/commit/52c1df6)) _(by Chris <chrisbreuer93@gmail.com>)_

### 🧹 Chores

- add release and buddy-bot workflows ([933a6e5](https://github.com/stacksjs/ts-webp/commit/933a6e5)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- set version to 0.0.1 ([3c627d2](https://github.com/stacksjs/ts-webp/commit/3c627d2)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- migrate to better-dx, add CI workflow ([5236d00](https://github.com/stacksjs/ts-webp/commit/5236d00)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- initial commit ([a5e3a3b](https://github.com/stacksjs/ts-webp/commit/a5e3a3b)) _(by Chris <chrisbreuer93@gmail.com>)_

### Contributors

- _Chris <chrisbreuer93@gmail.com>_
- _glennmichael123 <gtorregosa@gmail.com>_

### 🚀 Features

- **vp8l**: subtract-green + LZ77 + color cache → 5× compression on photos ([4c694bd](https://github.com/stacksjs/ts-webp/commit/4c694bd)) _(by Chris <chrisbreuer93@gmail.com>)_

### ⚡ Performance Improvements

- real binary heap, parallel-array tokens, two-level Huffman LUT ([52c1df6](https://github.com/stacksjs/ts-webp/commit/52c1df6)) _(by Chris <chrisbreuer93@gmail.com>)_

### 🧹 Chores

- add release and buddy-bot workflows ([933a6e5](https://github.com/stacksjs/ts-webp/commit/933a6e5)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- set version to 0.0.1 ([3c627d2](https://github.com/stacksjs/ts-webp/commit/3c627d2)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- migrate to better-dx, add CI workflow ([5236d00](https://github.com/stacksjs/ts-webp/commit/5236d00)) _(by glennmichael123 <gtorregosa@gmail.com>)_
- initial commit ([a5e3a3b](https://github.com/stacksjs/ts-webp/commit/a5e3a3b)) _(by Chris <chrisbreuer93@gmail.com>)_

### Contributors

- _Chris <chrisbreuer93@gmail.com>_
- _glennmichael123 <gtorregosa@gmail.com>_
