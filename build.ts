import { dts } from 'bun-plugin-dtsx'

await Bun.build({
  entrypoints: ['src/index.ts'],
  target: 'node',
  outdir: './dist',
  plugins: [dts()],
})

console.log('Build complete!')
