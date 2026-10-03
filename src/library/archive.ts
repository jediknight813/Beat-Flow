import type { Result } from '../engine/pipeline'
import type { SavedSong } from './storage'

export type ExportDetails = { title: string; artist: string; cover: Blob | null; coverChanged: boolean }

export function exportFileName(title: string): string {
  const printable = [...title].map((character) => character.charCodeAt(0) < 32 ? ' ' : character).join('')
  return `${printable.replace(/[\\/:*?"<>|]/g, ' ').replace(/\s+/g, ' ').trim() || 'Untitled'} - BeatFlow.zip`
}

export async function songFromResult(result: Result): Promise<SavedSong> {
  const { default: JSZip } = await import('jszip')
  const archive = await JSZip.loadAsync(await result.zip.arrayBuffer())
  const infoFile = archive.file('Info.dat')
  if (!infoFile) throw new Error('The generated map is missing Info.dat.')
  const info = JSON.parse(await infoFile.async('string'))
  const coverFile = typeof info._coverImageFilename === 'string' ? archive.file(info._coverImageFilename) : null
  const cover = coverFile ? new Blob([await coverFile.async('arraybuffer')], {
    type: /\.jpe?g$/i.test(info._coverImageFilename) ? 'image/jpeg' : 'image/png',
  }) : null
  const now = Date.now()
  return {
    zip: result.zip,
    details: {
      id: crypto.randomUUID(), title: result.detected.title, artist: result.detected.artist,
      duration: result.duration, bpm: result.detected.bpm,
      charts: result.charts.map((chart) => ({ difficulty: chart.difficulty, notes: chart.notes.length })),
      createdAt: now, updatedAt: now, cover,
      coverSource: result.detected.cover ? 'detected' : 'generated', fileName: result.fileName,
    },
  }
}

export async function updateSongExport(song: SavedSong, edits: ExportDetails): Promise<SavedSong> {
  const title = edits.title.trim()
  if (!title) throw new Error('Enter a song name before exporting.')
  const artist = edits.artist.trim()
  if (title === song.details.title && artist === song.details.artist && !edits.coverChanged) return song
  const { default: JSZip } = await import('jszip')
  const archive = await JSZip.loadAsync(await song.zip.arrayBuffer())
  const infoFile = archive.file('Info.dat')
  if (!infoFile) throw new Error('This map is missing Info.dat.')
  const info = JSON.parse(await infoFile.async('string'))
  const suffix = typeof info._songName === 'string' ? info._songName.match(/ \[AI [^\]]+\]$/)?.[0] ?? '' : ''
  info._songName = title + suffix
  info._songAuthorName = artist
  if (edits.coverChanged && edits.cover) {
    const previous = info._coverImageFilename
    if (typeof previous === 'string' && /\.(png|jpe?g|webp)$/i.test(previous)) archive.remove(previous)
    info._coverImageFilename = 'cover.png'
    archive.file('cover.png', await edits.cover.arrayBuffer())
  }
  archive.file('Info.dat', JSON.stringify(info, null, 2))
  const reportFile = archive.file('generation.json')
  if (reportFile) {
    const report = JSON.parse(await reportFile.async('string'))
    report.export = { title, artist, customCover: edits.coverChanged || song.details.coverSource === 'custom', editedAt: new Date().toISOString() }
    archive.file('generation.json', JSON.stringify(report, null, 2))
  }
  return {
    zip: await archive.generateAsync({ type: 'blob', compression: 'STORE' }),
    details: { ...song.details, title, artist, updatedAt: Date.now(), fileName: exportFileName(title),
      cover: edits.cover, coverSource: edits.coverChanged ? 'custom' : song.details.coverSource },
  }
}

export async function prepareCover(file: File): Promise<Blob> {
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) throw new Error('Choose a PNG, JPEG or WebP cover image.')
  if (file.size > 20 * 1024 * 1024) throw new Error('Choose a cover image smaller than 20 MB.')
  const bitmap = await createImageBitmap(file)
  try {
    const canvas = new OffscreenCanvas(512, 512)
    const context = canvas.getContext('2d')!
    const size = Math.min(bitmap.width, bitmap.height)
    context.drawImage(bitmap, (bitmap.width - size) / 2, (bitmap.height - size) / 2, size, size, 0, 0, 512, 512)
    return await canvas.convertToBlob({ type: 'image/png' })
  } finally { bitmap.close() }
}

export function downloadSong(song: SavedSong): void {
  const url = URL.createObjectURL(song.zip)
  const link = document.createElement('a')
  link.href = url
  link.download = song.details.fileName
  document.body.appendChild(link)
  link.click()
  link.remove()
  // Give the browser time to start reading the archive before releasing it.
  setTimeout(() => URL.revokeObjectURL(url), 30_000)
}
