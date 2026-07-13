import { describe, expect, it } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { decode } from '../src/index'

/**
 * VP8 + ALPH (lossy + alpha) round-trip tests.
 *
 * cwebp produces three flavours of the alpha plane depending on
 * `-alpha_method`:
 *   - method=0 (uncompressed): raw filter-encoded alpha bytes
 *   - method=1 (lossless): alpha encoded as a single-channel VP8L
 *     image whose green plane carries the values
 *
 * We exercise both via `-alpha_method 0` and the default (which picks
 * lossless), at multiple `-alpha_q` levels.
 */
function readPam(buf: Uint8Array): { width: number, height: number, data: Uint8Array } {
  let i = 0
  const lines: string[] = []
  while (i < buf.length) {
    let lineEnd = i
    while (lineEnd < buf.length && buf[lineEnd] !== 0x0A) lineEnd++
    const line = String.fromCharCode(...buf.slice(i, lineEnd))
    lines.push(line)
    i = lineEnd + 1
    if (line === 'ENDHDR') break
  }
  const width = Number(lines.find(l => l.startsWith('WIDTH'))!.split(' ')[1])
  const height = Number(lines.find(l => l.startsWith('HEIGHT'))!.split(' ')[1])
  return { width, height, data: buf.slice(i) }
}

describe('VP8 + ALPH (lossy with alpha)', () => {
  // `alpha-uncompressed` (method=0): bit-exact with dwebp.
  it('alpha-uncompressed matches dwebp bit-exact (RGBA)', () => {
    const fix = readFileSync(join(import.meta.dir, 'fixtures/alpha-uncompressed.webp'))
    const refRaw = readFileSync(join(import.meta.dir, 'fixtures/alpha-uncompressed-ref.pam'))
    const ref = readPam(new Uint8Array(refRaw))
    const out = decode(new Uint8Array(fix))
    expect(out.width).toBe(ref.width)
    expect(out.height).toBe(ref.height)
    expect(out.hasAlpha).toBe(true)
    let max = 0
    for (let i = 0; i < out.data.length; i++) {
      const d = Math.abs(out.data[i] - ref.data[i])
      if (d > max) max = d
    }
    expect(max).toBe(0)
  })

  // `alpha-q80` (method=1, lossless VP8L alpha plane, predictor + cache)
  // and `alpha-q95` (method=1 with a color-indexing transform): both were
  // long tracked as "divergent" — the actual cause was the length prefix
  // table in `vp8l/length.ts` using deflate's layout instead of VP8L's
  // prefix scheme, so any LZ77 copy with length code ≥ 4 misread its
  // extra bits and threw the whole stream out of alignment. With the
  // table corrected both decode bit-exact against dwebp.
  for (const name of ['alpha-q80', 'alpha-q95'] as const) {
    it(`${name} matches dwebp bit-exact (RGBA)`, () => {
      const fix = readFileSync(join(import.meta.dir, `fixtures/${name}.webp`))
      const refRaw = readFileSync(join(import.meta.dir, `fixtures/${name}-ref.pam`))
      const ref = readPam(new Uint8Array(refRaw))
      const out = decode(new Uint8Array(fix))
      expect(out.width).toBe(ref.width)
      expect(out.height).toBe(ref.height)
      expect(out.hasAlpha).toBe(true)
      let max = 0
      for (let i = 0; i < out.data.length; i++) {
        const d = Math.abs(out.data[i] - ref.data[i])
        if (d > max) max = d
      }
      expect(max).toBe(0)
    })
  }
})
