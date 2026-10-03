import { CanvasTexture, SRGBColorSpace } from 'three'

const angles = [0, Math.PI, -Math.PI / 2, Math.PI / 2, -Math.PI / 4, Math.PI / 4, (-3 * Math.PI) / 4, (3 * Math.PI) / 4]

function canvas(size: number) {
  const element = document.createElement('canvas')
  element.width = size
  element.height = size
  return [element, element.getContext('2d')!] as const
}

export function faceTexture(direction: number) {
  const [element, ctx] = canvas(256)
  // A narrow inset seam gives the colored face a defined edge without a black screen.
  ctx.strokeStyle = 'rgba(3, 4, 12, 0.5)'
  ctx.lineWidth = 5
  ctx.beginPath()
  ctx.roundRect(4, 4, 248, 248, 5)
  ctx.stroke()
  ctx.fillStyle = '#ffffff'
  ctx.translate(128, 128)
  if (direction === 8) {
    ctx.beginPath()
    ctx.arc(0, 0, 22, 0, Math.PI * 2)
    ctx.fill()
  } else {
    ctx.rotate(angles[direction] + Math.PI)
    ctx.beginPath()
    ctx.moveTo(-76, -82)
    ctx.lineTo(76, -82)
    ctx.lineTo(0, -38)
    ctx.closePath()
    ctx.fill()
  }
  const texture = new CanvasTexture(element)
  texture.colorSpace = SRGBColorSpace
  texture.anisotropy = 4
  return texture
}

export function sparkTexture() {
  const [element, ctx] = canvas(64)
  const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32)
  gradient.addColorStop(0, 'rgba(255,255,255,1)')
  gradient.addColorStop(0.35, 'rgba(255,255,255,0.7)')
  gradient.addColorStop(1, 'rgba(255,255,255,0)')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, 64, 64)
  const texture = new CanvasTexture(element)
  texture.colorSpace = SRGBColorSpace
  return texture
}
