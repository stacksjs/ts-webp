export { decodeAnimation } from './animation'
export { decode } from './decoder'
export { encode, encodeAsync, encodeWithAlpha } from './encoder'
export { encodeViaCwebp, hasCwebp } from './encoder-cli'
export { createRiffContainer, getWebpInfo, parseRiff } from './riff'
export type {
  RiffChunk,
  VP8FrameHeader,
  VP8LHeader,
  WebpAnimation,
  WebpAnimationFrame,
  WebpDecodeOptions,
  WebpEncodeOptions,
  WebpImageData,
  WebpInfo,
} from './types'

// Default export
import { decode } from './decoder'
import { encode } from './encoder'

export default {
  decode,
  encode,
}
