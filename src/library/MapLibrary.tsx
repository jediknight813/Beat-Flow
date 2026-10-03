import { BRAND_ICON, BRANDING_VERSION, COVER_BADGE_INSET, COVER_BADGE_SIZE, COVER_SIZE } from '../branding'
import { Fragment, useEffect, useRef, useState } from 'react'
import type { Result } from '../engine/pipeline'
import { downloadSong, prepareCover, songFromResult, updateSongExport, type ExportDetails } from './archive'
import { deleteSong, getSong, listSongs, saveSong, type SavedSong, type SongDetails } from './storage'

export type PanelView = 'generate' | 'export' | 'history'
const message = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong.'
const duration = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`

function Cover({ blob, title, branded = false }: { blob: Blob | null; title: string; branded?: boolean }) {
  const image = useRef<HTMLImageElement>(null)
  useEffect(() => {
    if (!blob || !image.current) return
    const next = URL.createObjectURL(blob)
    image.current.src = next
    return () => URL.revokeObjectURL(next)
  }, [blob])
  return <span className="cover-image">{blob ? <><img ref={image} alt={`${title} cover`} />{!branded && <img className="cover-brand" src={BRAND_ICON} alt="" aria-hidden="true" style={{ width: `${COVER_BADGE_SIZE / COVER_SIZE * 100}%`, height: `${COVER_BADGE_SIZE / COVER_SIZE * 100}%`, right: `${COVER_BADGE_INSET / COVER_SIZE * 100}%`, bottom: `${COVER_BADGE_INSET / COVER_SIZE * 100}%` }} />}</> : <span className="cover-placeholder" aria-hidden="true">♫</span>}</span>
}

function ExportEditor({ song, working, onExport }: { song: SavedSong; working: boolean; onExport: (edits: ExportDetails) => Promise<SavedSong | null> }) {
  const [title, setTitle] = useState(song.details.title)
  const [artist, setArtist] = useState(song.details.artist)
  const [cover, setCover] = useState(song.details.cover)
  const [coverChanged, setCoverChanged] = useState(false)
  const [coverBusy, setCoverBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const busy = working || coverBusy
  const dirty = title !== song.details.title || artist !== song.details.artist || coverChanged
  const changeCover = async (file?: File) => {
    if (!file) return
    setCoverBusy(true)
    setError(null)
    try { setCover(await prepareCover(file)); setCoverChanged(true) }
    catch (cause) { setError(message(cause)) }
    finally { setCoverBusy(false) }
  }
  return <form className="export-editor" aria-label="Export map" onSubmit={(event) => {
    event.preventDefault()
    if (!busy) void onExport({ title, artist, cover, coverChanged }).then((saved) => { if (saved) { setTitle(saved.details.title); setArtist(saved.details.artist); setCover(saved.details.cover); setCoverChanged(false) } })
  }}>
    <div className="panel-heading"><span>02 / EXPORT MAP</span><span className="export-ready">READY</span></div>
    <div className="export-artwork">
      <div className="export-artwork-glow" aria-hidden="true" />
      <label className={`export-cover${busy ? ' is-busy' : ''}`}>
        <Cover blob={cover} title={title || 'Song'} branded={!coverChanged && song.details.brandingVersion === BRANDING_VERSION} />
        <span className="export-cover-edit"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden="true"><path d="m10 3 3 3M3 13l3-1 7-7-2-2-7 7-1 3Z" /></svg>{coverBusy ? 'Loading…' : 'Change artwork'}</span>
        <input type="file" accept="image/png,image/jpeg,image/webp" disabled={busy} aria-label="Album cover"
          onChange={(event) => { void changeCover(event.target.files?.[0]); event.target.value = '' }} />
      </label>
    </div>
    <div className="export-song-details">
      <label className={`export-title-field${title.length > 26 ? ' is-long' : ''}`}><input aria-label="Song name" required maxLength={200} value={title} disabled={busy} onChange={(event) => setTitle(event.target.value)} /></label>
      <label className="export-artist-field"><input aria-label="Artist" maxLength={200} value={artist} placeholder="Artist name" disabled={busy} onChange={(event) => setArtist(event.target.value)} /></label>
      <p className="export-song-stats">{duration(song.details.duration)}<span>·</span>{Math.round(song.details.bpm)} BPM</p>
    </div>
    <div className="export-charts">{song.details.charts.map((chart) => <span key={chart.difficulty}>{chart.difficulty === 'ExpertPlus' ? 'Expert+' : chart.difficulty} <small>{chart.notes.toLocaleString()} notes</small></span>)}</div>
    {error && <p className="error-message" role="alert">{error}</p>}
    <button type="submit" className="generate-button" disabled={busy || !title.trim()}><span>{working ? 'Preparing export…' : 'Download map'}</span><span aria-hidden="true">↓</span></button>
    {!working && dirty && <p className="library-save-status" role="status">Changes will be saved when you export</p>}
  </form>
}

type Props = { result: Result | null; view: PanelView; onViewChange: (view: PanelView) => void; onHistoryChange: (count: number) => void; generationBusy: boolean }

export default function MapLibrary({ result, view, onViewChange, onHistoryChange, generationBusy }: Props) {
  const [songs, setSongs] = useState<SongDetails[]>([])
  const [active, setActive] = useState<SavedSong | null>(null)
  const [working, setWorking] = useState(false)
  const [processedResult, setProcessedResult] = useState<Result | null>(null)
  const preparing = result !== null && processedResult !== result
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [removeId, setRemoveId] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void listSongs().then((items) => { if (!cancelled) setSongs(items) })
      .catch((cause: unknown) => { if (!cancelled) setError(`History unavailable: ${message(cause)}`) })
      .finally(() => { if (!cancelled) setLoadingHistory(false) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => { onHistoryChange(songs.length) }, [songs, onHistoryChange])

  // The navbar owns view switching: reset when the view changes, and reload history on entry.
  const [shownView, setShownView] = useState(view)
  if (view !== shownView) {
    setShownView(view)
    setError(null)
    if (view === 'generate') { setActive(null); setProcessedResult(null) }
    if (view === 'history') setLoadingHistory(true)
  }
  useEffect(() => {
    if (view !== 'history') return
    let cancelled = false
    void listSongs().then((items) => { if (!cancelled) setSongs(items) })
      .catch((cause: unknown) => { if (!cancelled) setError(`History unavailable: ${message(cause)}`) })
      .finally(() => { if (!cancelled) setLoadingHistory(false) })
    return () => { cancelled = true }
  }, [view])

  useEffect(() => {
    if (!result) return
    let cancelled = false
    void (async () => {
      try {
        const song = await songFromResult(result)
        if (cancelled) return
        setError(null)
        setActive(song)
        try {
          await saveSong(song)
          if (cancelled) return
          setSongs(await listSongs())
        } catch {
          if (!cancelled) setError('Could not save this map to history. You can still download it below.')
        }
      } catch (cause) {
        if (!cancelled) { setActive(null); setError(`Could not open the export editor: ${message(cause)}`) }
      } finally { if (!cancelled) setProcessedResult(result) }
    })()
    return () => { cancelled = true }
  }, [result])

  const exportMap = async (edits: ExportDetails): Promise<SavedSong | null> => {
    if (!active || working) return null
    setWorking(true)
    setError(null)
    try {
      const updated = await updateSongExport(active, edits)
      setActive(updated)
      try { await saveSong(updated); setSongs(await listSongs()) }
      catch { setError('Download is ready, but these edits could not be saved to history. Browser storage may be full.') }
      downloadSong(updated)
      return updated
    } catch (cause) { setError(message(cause)); return null }
    finally { setWorking(false) }
  }

  const openSong = async (id: string, downloadOnly = false) => {
    if (working || generationBusy) return
    setWorking(true)
    setError(null)
    try {
      const original = await getSong(id)
      const song = await updateSongExport(original, { title: original.details.title, artist: original.details.artist, cover: original.details.cover, coverChanged: false })
      if (downloadOnly) {
        if (song !== original) {
          try { await saveSong(song); setSongs(await listSongs()) }
          catch { setError('Download is ready, but these updates could not be saved to history.') }
        }
        downloadSong(song)
      } else { setActive(song); onViewChange('export') }
    } catch (cause) { setError(message(cause)) }
    finally { setWorking(false) }
  }

  const removeSong = async (id: string) => {
    setWorking(true)
    setError(null)
    try { await deleteSong(id); setSongs(await listSongs()); setRemoveId(null) }
    catch (cause) { setError(message(cause)) }
    finally { setWorking(false) }
  }

  return <>
    {view === 'export' && <>
      {preparing ? <p className="library-message" role="status">Preparing your export…</p>
        : active ? <ExportEditor key={active.details.id} song={active} working={working} onExport={exportMap} />
        : working ? <p className="library-message" role="status">Preparing your export…</p>
          : result && <button type="button" className="generate-button" onClick={() => {
            const link = document.createElement('a'); const url = URL.createObjectURL(result.zip)
            link.href = url; link.download = result.fileName; link.click(); setTimeout(() => URL.revokeObjectURL(url), 30_000)
          }}>Download original map</button>}
    </>}
    {view === 'history' && <section className="song-history" aria-label="Generated songs">
      <div className="panel-heading"><span>YOUR MAPS</span><span className="file-label">SAVED IN THIS BROWSER</span></div>
      {loadingHistory ? <p className="library-message">Loading history…</p> : songs.length === 0 ? <p className="library-message">Your generated maps will appear here.</p> : <ul>
        {/* Batch tracks share an albumId; singles keep their own row. */}
        {(() => {
          const groups: { albumId: string | null; songs: SongDetails[] }[] = []
          const byAlbum = new Map<string, { albumId: string | null; songs: SongDetails[] }>()
          for (const song of songs) {
            const existing = song.albumId ? byAlbum.get(song.albumId) : undefined
            if (existing) { existing.songs.push(song); continue }
            const group = song.albumId ? { albumId: song.albumId, songs: [song] } : { albumId: null, songs: [song] }
            if (song.albumId) byAlbum.set(song.albumId, group)
            groups.push(group)
          }
          return groups.map((group) => <Fragment key={group.albumId ?? group.songs[0].id}>
            {group.albumId && <li className="history-album-heading">
              <span>Album · {group.songs.length} map{group.songs.length === 1 ? '' : 's'}</span>
              <span>{new Date(group.songs[0].createdAt).toLocaleDateString()}</span>
            </li>}
            {group.songs.map((song) => <li key={song.id} className={group.albumId ? 'in-album' : undefined}>
              <div className="history-cover"><Cover blob={song.cover} title={song.title} branded={song.brandingVersion === BRANDING_VERSION} /></div>
              <div className="history-song"><strong>{song.title}</strong><span>{song.artist || 'Unknown artist'}</span><small>{duration(song.duration)}{song.version ? ` · ${song.version}` : ''} · {new Date(song.createdAt).toLocaleDateString()}</small>
                <div className="history-actions">
                  <button type="button" disabled={working} onClick={() => void openSong(song.id)}>Edit / export</button>
                  <button type="button" disabled={working} onClick={() => void openSong(song.id, true)}>Download</button>
                  <button type="button" disabled={working} className="history-remove" aria-label={`Remove ${song.title} from history`} onClick={() => setRemoveId(song.id)}>Remove</button>
                </div>
                {removeId === song.id && <div className="history-confirm"><span>Remove this saved map?</span><button type="button" disabled={working} onClick={() => void removeSong(song.id)}>Remove</button><button type="button" onClick={() => setRemoveId(null)}>Cancel</button></div>}
              </div>
            </li>)}
          </Fragment>)
        })()}
      </ul>}
    </section>}
    {error && view !== 'generate' && <p className="error-message" role="alert">{error}</p>}
  </>
}
