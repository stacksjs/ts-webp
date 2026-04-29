import { describe, expect, it } from 'bun:test'
import { BoolDecoder } from '../src/vp8/bool-decoder'
import { parseVP8Header } from '../src/vp8/header'

describe('VP8 boolean decoder', () => {
  it('readLiteral round-trips through encode→decode', () => {
    // Hand-construct a stream where two known literals are encoded with
    // probability 128 (i.e. uniform), then verify the decoder reads
    // them back. With prob=128 the boolean-arithmetic coder reduces to
    // straight bit emission (one input bit per output bit), so we can
    // synthesise the wire form by writing the bits directly.
    //
    // Bits to encode (MSB first, as readLiteral reads): 1011 + 010
    //   first literal (4 bits): 0b1011 = 11
    //   second literal (3 bits): 0b010 = 2
    //
    // With p=128 the encoder splits the [0..range) line in half each
    // time and writes the high bits straight to the stream. The byte
    // stream is therefore the bits packed MSB-first into the first
    // byte: `1011 010_` = 0b10110100 = 0xB4 (last bit doesn't matter).
    const data = new Uint8Array([0xB4, 0x00])
    const bool = new BoolDecoder(data)
    expect(bool.readLiteral(4)).toBe(11)
    expect(bool.readLiteral(3)).toBe(2)
  })

  it('readSignedLiteral handles positive and negative values', () => {
    // Encode (mag=5, sign=1) → -5, then (mag=3, sign=0) → 3. With
    // p=128 the wire is just MSB-first bit packing.
    //
    // First value: 5 in 4 bits = 0b0101, then sign bit = 1
    // Second: 3 in 4 bits = 0b0011, then sign bit = 0
    // Concatenated: 01011 00110 = 01011001 10000000 = 0x59 0x80
    const data = new Uint8Array([0x59, 0x80, 0x00])
    const bool = new BoolDecoder(data)
    expect(bool.readSignedLiteral(4)).toBe(-5)
    expect(bool.readSignedLiteral(4)).toBe(3)
  })

  it('handles empty input without crashing', () => {
    // Pulling from an empty stream is well-defined: reads return
    // whatever the priming put in `value` (which is 0), then keep
    // returning 0 forever. Real corrupted bitstreams hit this path;
    // the decoder must not loop or throw spuriously.
    const bool = new BoolDecoder(new Uint8Array(0))
    expect(bool.readLiteral(4)).toBe(0)
  })
})

describe('VP8 frame header parser', () => {
  /** Build a minimal valid VP8 frame header with no segmentation, no LF adj. */
  function makeMinimalKeyframeHeader(width: number, height: number, partitions = 1): Uint8Array {
    // Frame tag: keyframe, version=0, show_frame=1, partition size = 32 bytes.
    // Layout: bit 0 = keyframe (0 = keyframe), 1..3 = version, 4 = show_frame,
    //         5..23 = first_partition_size.
    // Use a larger first-partition size so the boolean coder has enough
    // headroom to consume all the proba-update flag bits at near-zero cost.
    const partitionSize = 256
    const frameTag = (partitionSize << 5) | (1 << 4) | (0 << 1) | 0
    const out = new Uint8Array(10 + partitionSize + 32)
    out[0] = frameTag & 0xFF
    out[1] = (frameTag >> 8) & 0xFF
    out[2] = (frameTag >> 16) & 0xFF
    out[3] = 0x9D
    out[4] = 0x01
    out[5] = 0x2A
    // width and height in the lower 14 bits; scale = 0
    out[6] = width & 0xFF
    out[7] = (width >> 8) & 0xFF
    out[8] = height & 0xFF
    out[9] = (height >> 8) & 0xFF
    // Partition payload is bool-coded. With probability 128 each, every
    // bit becomes a straight MSB-first bit on the wire. Bits to write:
    //   color_space = 0
    //   clamp = 0
    //   segmentation_enabled = 0
    //   filter_type = 0
    //   filter_level (6 bits) = 0
    //   sharpness (3 bits) = 0
    //   loop_filter_adj_enabled = 0
    //   num_token_partitions log2 (2 bits) = log2(partitions)
    //   yac_qi (7 bits) = 64
    //   y_dc_delta_present = 0  (skip the 5 bits that would follow)
    //   y2_dc_delta_present = 0
    //   y2_ac_delta_present = 0
    //   uv_dc_delta_present = 0
    //   uv_ac_delta_present = 0
    //   refresh_golden = 1
    //   refresh_altref = 1
    //   sign_bias_golden = 0
    //   sign_bias_altref = 0
    //   refresh_entropy_probs = 0
    //   refresh_last = 1
    //
    // Total bits = 4 + 1 + 6 + 3 + 1 + 2 + 7 + 5 + 1 + 1 + 1 + 1 + 1 + 1 = 35 bits.
    // We pack them into the partition area starting at offset 10.
    const log2Parts = Math.round(Math.log2(partitions))
    const bits: number[] = []
    bits.push(0, 0, 0, 0) // color_space, clamp, seg_enabled, filter_type
    for (let i = 5; i >= 0; i--) bits.push(0) // filter_level (6 bits)
    for (let i = 2; i >= 0; i--) bits.push(0) // sharpness (3 bits)
    bits.push(0) // loop_filter_adj_enabled
    bits.push((log2Parts >> 1) & 1, log2Parts & 1) // partition count, MSB-first
    for (let i = 6; i >= 0; i--) bits.push((64 >> i) & 1) // yac_qi=64
    for (let i = 0; i < 5; i++) bits.push(0) // 5 dc/ac delta presence flags
    bits.push(1, 1, 0, 0, 0, 1) // refresh flags + sign biases

    // Pack bits MSB-first into bytes.
    for (let i = 0; i < bits.length; i++) {
      const byteIdx = 10 + (i >> 3)
      out[byteIdx] |= bits[i] << (7 - (i & 7))
    }
    return out
  }

  it('parses a minimal valid keyframe', () => {
    const buf = makeMinimalKeyframeHeader(100, 50)
    const h = parseVP8Header(buf)
    expect(h.frame.keyframe).toBe(true)
    expect(h.frame.width).toBe(100)
    expect(h.frame.height).toBe(50)
    expect(h.frame.showFrame).toBe(true)
    expect(h.numPartitions).toBe(1)
    expect(h.colorSpace).toBe(0)
    expect(h.segmentation.useSegment).toBe(false)
  })

  it('captures the partition count', () => {
    for (const p of [1, 2, 4, 8]) {
      const h = parseVP8Header(makeMinimalKeyframeHeader(16, 16, p))
      expect(h.numPartitions).toBe(p)
    }
  })

  it('throws on missing start code', () => {
    const buf = makeMinimalKeyframeHeader(16, 16)
    buf[3] = 0x00 // corrupt the start code byte
    expect(() => parseVP8Header(buf)).toThrow(/start code/)
  })

  it('throws when chunk is too short', () => {
    expect(() => parseVP8Header(new Uint8Array(5))).toThrow(/shorter/)
  })

  it('throws on inter-frame (only keyframes are supported in WebP)', () => {
    const buf = makeMinimalKeyframeHeader(16, 16)
    buf[0] |= 0x01 // set the keyframe bit (1 = NOT a keyframe in spec language)
    expect(() => parseVP8Header(buf)).toThrow(/keyframe/i)
  })
})
