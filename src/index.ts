export { decode } from './decoder'
export { encode, encodeWithAlpha } from './encoder'
export { parseRiff, getWebpInfo, createRiffContainer } from './riff'
export type {
  WebpImageData,
  WebpEncodeOptions,
  WebpDecodeOptions,
  WebpInfo,
  RiffChunk,
  VP8FrameHeader,
  VP8LHeader,
} from './types'

// Default export
import { decode } from './decoder'
import { encode } from './encoder'

export default {
  decode,
  encode,
}
