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

/**
 * One frame of an animated WebP, with positioning and timing metadata
 * pulled from the surrounding ANMF chunk header.
 */
export interface WebpAnimationFrame {
  /** Decoded RGBA pixel data for the frame's bounding box. */
  image: WebpImageData
  /** X offset (in pixels) where the frame is composited onto the canvas. */
  x: number
  /** Y offset where the frame is composited onto the canvas. */
  y: number
  /** Display duration of the frame in milliseconds. */
  duration: number
  /**
   * Blending mode for compositing this frame onto the canvas:
   *   - 'overlay' = alpha-composite over previous frame
   *   - 'replace' = overwrite previous frame's pixels in this rectangle
   */
  blend: 'overlay' | 'replace'
  /**
   * What to do with the canvas after this frame is shown:
   *   - 'none'       = leave the canvas as-is for the next frame
   *   - 'background' = restore the rectangle to the background colour
   */
  dispose: 'none' | 'background'
}

/**
 * Decoded animation, with the canvas geometry, loop count, and an
 * ordered list of frames.
 */
export interface WebpAnimation {
  /** Canvas width (pixels). */
  width: number
  /** Canvas height (pixels). */
  height: number
  /**
   * Number of times to loop the animation. 0 = infinite, the default
   * for most content. Other values mean play exactly N times.
   */
  loopCount: number
  /** Background ARGB colour as a packed `(A << 24) | (R << 16) | (G << 8) | B`. */
  backgroundColor: number
  /** Animation frames in display order. */
  frames: WebpAnimationFrame[]
}
