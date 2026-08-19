import { dts } from 'bun-plugin-dtsx'

await Bun.build({
  minify: true,
  entrypoints: ['src/index.ts'],
  target: 'bun',
  outdir: './dist',
  plugins: [dts()],
})
