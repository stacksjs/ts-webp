import { describe, expect, it } from 'bun:test'
import { writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawnSync } from 'bun'
import { createRiffContainer } from '../src/riff'
import { encodeVP8 } from '../src/vp8/encoder'

/**
 * VP8 encoder × dwebp interop. Verifies that the bitstream we emit
 * parses cleanly through libwebp's reference decoder. We don't compare
 * pixel values bit-exactly — that requires matching cwebp's mode-search
 * heuristics — but if `dwebp` decodes our output without error and
 * produces an image of the right size, the bitstream is structurally
 * correct.
 *
 * Skipped silently if `dwebp` isn't on PATH (e.g. CI without libwebp
 * installed).
 */

function makeImage(width: number, height: number, fill: (x: number, y: number) => [number, number, number, number]) {
  const data = new Uint8Array(width * height * 4)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b, a] = fill(x, y)
      const i = (y * width + x) * 4
      data[i] = r
      data[i + 1] = g
      data[i + 2] = b
      data[i + 3] = a
    }
  }
  return { data, width, height, hasAlpha: false }
}

const dwebpAvailable = (() => {
  try {
    const r = spawnSync(['which', 'dwebp'])
    return r.exitCode === 0
  }
  catch { return false }
})()

describe.skipIf(!dwebpAvailable)('VP8 lossy encoder × dwebp interop', () => {
  it('dwebp can decode a 16×16 solid-colour VP8 we encode', async () => {
    const img = makeImage(16, 16, () => [120, 80, 60, 255])
    const vp8Chunk = encodeVP8(img, { quality: 60 })
    const webp = createRiffContainer([
      { fourCC: 'VP8 ', data: vp8Chunk },
    ])
    const path = join(tmpdir(), `ts-webp-vp8-test-${Date.now()}.webp`)
    writeFileSync(path, webp)
    const r = spawnSync(['dwebp', '-pam', path, '-o', '-'])
    expect(r.exitCode).toBe(0)
    // dwebp -pam emits a header followed by raw RGB(A) bytes; we just
    // assert it didn't error and produced output of plausible size.
    const stdout = await new Response(r.stdout).bytes()
    expect(stdout.length).toBeGreaterThan(16 * 16 * 3) // header + ≥3 bytes/pixel
  })

  it('dwebp can decode a 32×32 gradient VP8 we encode', async () => {
    const img = makeImage(32, 32, x => [(x * 8) & 0xFF, 128, 200, 255])
    const vp8Chunk = encodeVP8(img, { quality: 50 })
    const webp = createRiffContainer([
      { fourCC: 'VP8 ', data: vp8Chunk },
    ])
    const path = join(tmpdir(), `ts-webp-vp8-test2-${Date.now()}.webp`)
    writeFileSync(path, webp)
    const r = spawnSync(['dwebp', '-pam', path, '-o', '-'])
    expect(r.exitCode).toBe(0)
    const stdout = await new Response(r.stdout).bytes()
    expect(stdout.length).toBeGreaterThan(32 * 32 * 3)
  })
})
