export type Tags = {
  title: string | null
  artist: string | null
  cover: { data: Uint8Array; mime: string } | null
}

const latin1 = new TextDecoder('latin1')
const utf8 = new TextDecoder('utf-8')

function syncsafe(b: Uint8Array, o: number): number {
  return ((b[o] & 0x7f) << 21) | ((b[o + 1] & 0x7f) << 14) | ((b[o + 2] & 0x7f) << 7) | (b[o + 3] & 0x7f)
}

function u32(b: Uint8Array, o: number): number {
  return ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0
}

function u24(b: Uint8Array, o: number): number {
  return (b[o] << 16) | (b[o + 1] << 8) | b[o + 2]
}

function ascii(b: Uint8Array, o: number, n: number): string {
  return String.fromCharCode(...b.subarray(o, o + n))
}

function utf16(b: Uint8Array, bigEndian: boolean): string {
  let s = ''
  for (let i = 0; i + 1 < b.length; i += 2) s += String.fromCharCode(bigEndian ? (b[i] << 8) | b[i + 1] : b[i] | (b[i + 1] << 8))
  return s
}

function decodeText(enc: number, b: Uint8Array): string {
  let s: string
  if (enc === 1) {
    const be = b[0] === 0xfe && b[1] === 0xff
    const bom = (b[0] === 0xff && b[1] === 0xfe) || be
    s = utf16(bom ? b.subarray(2) : b, be)
  } else if (enc === 2) s = utf16(b, true)
  else if (enc === 3) s = utf8.decode(b)
  else s = latin1.decode(b)
  return s.split('\0').filter(Boolean)[0]?.trim() ?? ''
}

function terminator(b: Uint8Array, from: number, enc: number): number {
  const wide = enc === 1 || enc === 2
  for (let i = from; i < b.length; i += wide ? 2 : 1) {
    if (b[i] === 0 && (!wide || b[i + 1] === 0)) return i + (wide ? 2 : 1)
  }
  return b.length
}

function unsync(b: Uint8Array): Uint8Array {
  const out: number[] = []
  for (let i = 0; i < b.length; i++) {
    out.push(b[i])
    if (b[i] === 0xff && b[i + 1] === 0) i++
  }
  return Uint8Array.from(out)
}

function picture(body: Uint8Array, v2: boolean): Tags['cover'] {
  const enc = body[0]
  let o = 1
  let mime: string
  if (v2) {
    const fmt = ascii(body, 1, 3).toLowerCase()
    mime = fmt === 'png' ? 'image/png' : 'image/jpeg'
    o = 4
  } else {
    const end = body.indexOf(0, 1)
    mime = latin1.decode(body.subarray(1, end)).toLowerCase() || 'image/jpeg'
    o = end + 1
  }
  o = terminator(body, o + 1, enc)
  const data = body.slice(o)
  if (!data.length) return null
  return { data, mime: mime.includes('png') ? 'image/png' : 'image/jpeg' }
}

function id3(b: Uint8Array): Tags | null {
  if (b.length < 10 || ascii(b, 0, 3) !== 'ID3') return null
  const major = b[3]
  const flags = b[5]
  let tag = b.subarray(10, 10 + syncsafe(b, 6))
  if (flags & 0x80 && major < 4) tag = unsync(tag)
  let o = 0
  if (flags & 0x40) o = major === 4 ? syncsafe(tag, 0) : u32(tag, 0) + 4
  const out: Tags = { title: null, artist: null, cover: null }
  const v2 = major === 2
  const head = v2 ? 6 : 10
  while (o + head <= tag.length) {
    const id = ascii(tag, o, v2 ? 3 : 4)
    if (!/^[A-Z0-9]{3,4}$/.test(id)) break
    const size = v2 ? u24(tag, o + 3) : major === 4 ? syncsafe(tag, o + 4) : u32(tag, o + 4)
    const frameFlags = v2 ? 0 : tag[o + 9]
    let body = tag.subarray(o + head, o + head + size)
    o += head + size
    if (major === 4 && frameFlags & 0x02) body = unsync(body)
    if (major === 4 && frameFlags & 0x01) body = body.subarray(4)
    if (id === 'TIT2' || id === 'TT2') out.title ??= decodeText(body[0], body.subarray(1)) || null
    else if (id === 'TPE1' || id === 'TP1') out.artist ??= decodeText(body[0], body.subarray(1)) || null
    else if ((id === 'APIC' || id === 'PIC') && !out.cover) out.cover = picture(body, v2)
  }
  return out
}

type Atom = { type: string; start: number; end: number }

function atoms(b: Uint8Array, start: number, end: number): Atom[] {
  const out: Atom[] = []
  let o = start
  while (o + 8 <= end) {
    let size = u32(b, o)
    let head = 8
    if (size === 1) {
      size = u32(b, o + 8) * 2 ** 32 + u32(b, o + 12)
      head = 16
    } else if (size === 0) size = end - o
    if (size < head || o + size > end) break
    out.push({ type: latin1.decode(b.subarray(o + 4, o + 8)), start: o + head, end: o + size })
    o += size
  }
  return out
}

function child(b: Uint8Array, parent: Atom | undefined, type: string, skip = 0): Atom | undefined {
  return parent ? atoms(b, parent.start + skip, parent.end).find((a) => a.type === type) : undefined
}

function mp4(b: Uint8Array): Tags | null {
  if (b.length < 12 || ascii(b, 4, 4) !== 'ftyp') return null
  const moov = atoms(b, 0, b.length).find((a) => a.type === 'moov')
  const meta = child(b, child(b, moov, 'udta'), 'meta')
  const ilst = child(b, meta, 'ilst', 4) ?? child(b, meta, 'ilst')
  const out: Tags = { title: null, artist: null, cover: null }
  if (!ilst) return out
  for (const item of atoms(b, ilst.start, ilst.end)) {
    const data = child(b, item, 'data')
    if (!data || data.end - data.start < 8) continue
    const kind = u32(b, data.start) & 0xffffff
    const payload = b.subarray(data.start + 8, data.end)
    if (item.type === '©nam') out.title ??= utf8.decode(payload).trim() || null
    else if (item.type === '©ART' || (item.type === 'aART' && !out.artist)) out.artist ??= utf8.decode(payload).trim() || null
    else if (item.type === 'covr' && !out.cover && payload.length) out.cover = { data: payload.slice(), mime: kind === 14 ? 'image/png' : 'image/jpeg' }
  }
  return out
}

export function fileTags(bytes: Uint8Array): Tags {
  try {
    return id3(bytes) ?? mp4(bytes) ?? { title: null, artist: null, cover: null }
  } catch {
    return { title: null, artist: null, cover: null }
  }
}

export function nameTags(fileName: string): { title: string; artist: string } {
  const base = fileName.replace(/\.[^.]+$/, '').replace(/_/g, ' ').replace(/^\d{1,3}(-\d{1,3})?[\s.-]+/, '').trim()
  const parts = base.split(' - ')
  if (parts.length >= 2) return { artist: parts[0].trim(), title: parts.slice(1).join(' - ').trim() }
  return { title: base || 'Untitled', artist: '' }
}
