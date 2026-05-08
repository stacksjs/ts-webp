import type { WebpEncodeOptions, WebpImageData } from './types'

/**
 * Shell out to the system `cwebp` binary (libwebp) for real WebP
 * encoding. Returns the encoded bytes, or `null` if the binary
 * isn't available or fails to run.
 *
 * Why bother shelling out? The bundled VP8 lossy encoder in this
 * package is "minimal-but-functional" — single-segment, 16×16
 * intra-DC prediction, no mode search, no rate-control loop — so
 * the output is competitively *correct* but not competitively
 * *small*. For real-world traffic (catalog photos, OG images, the
 * whole point of WebP) you want libwebp's R/D loop. Shelling out
 * is the cheap path: any modern dev machine and most CI runners
 * already have `cwebp` (Homebrew `webp`, apt `webp`, etc.).
 *
 * The transport is a 4-byte BE width + 4-byte BE height + raw RGBA
 * payload, fed to `cwebp` over stdin via the `-` (stdin) form. We
 * read the encoded bytes back from stdout. No temp files.
 *
 * Why not write a PNG and pipe that? `cwebp` accepts PNG/JPEG/TIFF/
 * WebP/PAM/PPM/PGM; the simplest format we can synthesize with no
 * dependencies is PAM (P7 / portable arbitrary map), which is just
 * a header + raw RGBA. That's what we use here — it lets `cwebp`
 * decode our exact pixels without us having to ship a PNG encoder
 * just to talk to it.
 */
export async function encodeViaCwebp(
  imageData: WebpImageData,
  options: WebpEncodeOptions,
): Promise<Uint8Array | null> {
  const bin = options.cwebpPath || 'cwebp'

  const { width, height, data } = imageData
  if (data.byteLength !== width * height * 4)
    throw new Error('ts-webp: imageData.data must be RGBA (width × height × 4 bytes)')

  // Build PAM (P7) header. PAM is the simplest "raw RGBA + small
  // ASCII header" format any libnetpbm / libimageio reader (libwebp
  // included) will accept. cwebp falls through to the imageio path
  // when the input is PAM.
  const header = `P7\nWIDTH ${width}\nHEIGHT ${height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n`
  const headerBytes = new TextEncoder().encode(header)
  const stdin = new Uint8Array(headerBytes.byteLength + data.byteLength)
  stdin.set(headerBytes, 0)
  stdin.set(data, headerBytes.byteLength)

  // cwebp args:
  //   -q <0..100>       quality (lossy)
  //   -lossless         force lossless path
  //   -m <0..6>         method/effort (0 fastest, 6 slowest+smallest)
  //   -alpha_q <0..100> alpha plane quality
  //   -metadata none    strip everything; we don't surface metadata
  //   -quiet            suppress progress noise on stderr
  //   -o -              write to stdout
  //   --                end of options (next arg = input)
  //   -                 input from stdin
  const args: string[] = []
  if (options.lossless) args.push('-lossless')
  else args.push('-q', String(options.quality ?? 80))
  if (typeof options.effort === 'number')
    args.push('-m', String(Math.max(0, Math.min(6, options.effort))))
  args.push('-alpha_q', '100', '-metadata', 'none', '-quiet', '-o', '-', '--', '-')

  let proc: ReturnType<typeof Bun.spawn>
  try {
    proc = Bun.spawn([bin, ...args], {
      stdin: 'pipe',
      stdout: 'pipe',
      stderr: 'pipe',
    })
  }
  catch {
    // ENOENT — `cwebp` not on PATH. Caller falls back to pure-ts.
    return null
  }

  // Pipe input then close so cwebp knows the stream is done.
  // proc.stdin is typed as `number | FileSink` due to the union of stdio
  // option types; with `stdin: 'pipe'` it's the FileSink-shaped object.
  const sink = proc.stdin as { getWriter?: () => { write: (b: Uint8Array) => Promise<void>, close: () => Promise<void> }, write?: (b: Uint8Array) => void, end?: () => void } | undefined
  const writer = sink?.getWriter?.()
  if (writer) {
    await writer.write(stdin)
    await writer.close()
  }
  else {
    sink?.write?.(stdin)
    sink?.end?.()
  }

  const [out, exitCode] = await Promise.all([
    new Response(proc.stdout as ReadableStream<Uint8Array>).bytes(),
    proc.exited,
  ])

  if (exitCode !== 0) {
    // Not an exception — caller can fall back to pure-ts on its own.
    return null
  }

  return out
}

/**
 * Best-effort detection: does `cwebp` exist on PATH? Cached after
 * the first probe so a busy script doesn't spawn a probe child for
 * every encode.
 */
let cwebpAvailable: boolean | null = null
export async function hasCwebp(binPath?: string): Promise<boolean> {
  if (binPath) {
    // Custom path — probe directly without caching the result, since
    // a per-call override shouldn't poison the global cache.
    try {
      const proc = Bun.spawn([binPath, '-version'], { stdout: 'ignore', stderr: 'ignore' })
      return (await proc.exited) === 0
    }
    catch {
      return false
    }
  }
  if (cwebpAvailable !== null) return cwebpAvailable
  try {
    const proc = Bun.spawn(['cwebp', '-version'], { stdout: 'ignore', stderr: 'ignore' })
    cwebpAvailable = (await proc.exited) === 0
  }
  catch {
    cwebpAvailable = false
  }
  return cwebpAvailable
}
