// The same mark and placement are used by the live preview and exported cover.
export const BRANDING_VERSION = 1
export const BRAND_ICON = `${(import.meta.env?.BASE_URL ?? '/')}icon.svg`
export const COVER_SIZE = 512
export const COVER_BADGE_SIZE = 64
export const COVER_BADGE_INSET = 16

export function songTitle(title: string): string {
  return title.trim().replace(/(?:\s*\[(?:BF|AI [^\]]+)\])+$/gi, '').trim()
}

export function brandedSongTitle(title: string): string {
  return `${songTitle(title) || 'Untitled'} [BF]`
}

let icon: Promise<HTMLImageElement> | null = null
function brandIcon(): Promise<HTMLImageElement> {
  if (!icon) {
    const image = new Image()
    image.src = BRAND_ICON
    icon = image.decode().then(() => image).catch((error: unknown) => { icon = null; throw error })
  }
  return icon
}

export async function brandCover(cover: Blob): Promise<Blob> {
  const bitmap = await createImageBitmap(cover)
  try {
    const canvas = new OffscreenCanvas(COVER_SIZE, COVER_SIZE)
    const context = canvas.getContext('2d')
    if (!context) throw new Error('Unable to prepare the album artwork.')
    const side = Math.min(bitmap.width, bitmap.height)
    context.drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, COVER_SIZE, COVER_SIZE)
    const corner = COVER_SIZE - COVER_BADGE_SIZE - COVER_BADGE_INSET
    context.drawImage(await brandIcon(), corner, corner, COVER_BADGE_SIZE, COVER_BADGE_SIZE)
    return await canvas.convertToBlob({ type: 'image/png' })
  } finally { bitmap.close() }
}
