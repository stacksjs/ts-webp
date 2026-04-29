/**
 * WebP image data
 */
export interface WebpImageData {
  /** Pixel data in RGBA format (4 bytes per pixel) */
  data: Uint8Array
  /** Image width in pixels */
  width: number
  /** Image height in pixels */
  height: number
  /** Whether the image has an alpha channel */
  hasAlpha?: boolean
}

/**
 * WebP encoding options
 */
export interface WebpEncodeOptions {
  /** Quality for lossy encoding (0-100, default: 80) */
  quality?: number
  /** Use lossless encoding */
  lossless?: boolean
  /** Compression effort (0-9, default: 6) */
  effort?: number
  /** Enable alpha channel */
  alpha?: boolean
}

/**
 * WebP decoding options
 */
export interface WebpDecodeOptions {
  /** Output format (RGBA or RGB) */
  format?: 'rgba' | 'rgb'
}

/**
 * RIFF chunk structure
 */
export interface RiffChunk {
  fourCC: string
  size: number
  data: Uint8Array
  offset: number
}

/**
 * WebP file info
 */
export interface WebpInfo {
  width: number
  height: number
  hasAlpha: boolean
  isLossless: boolean
  hasAnimation: boolean
  isExtended: boolean
}

/**
 * VP8 frame header
 */
export interface VP8FrameHeader {
  keyframe: boolean
  version: number
  showFrame: boolean
  firstPartSize: number
  width: number
  height: number
  xScale: number
  yScale: number
}

/**
 * VP8L image header
 */
export interface VP8LHeader {
  width: number
  height: number
  hasAlpha: boolean
  version: number
}
