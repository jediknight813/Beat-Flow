import { useEffect, useRef, useState } from 'react'
import type { Result } from '../engine/pipeline'
import { downloadSong, prepareCover, songFromResult, updateSongExport, type ExportDetails } from './archive'
import { deleteSong, getSong, listSongs, saveSong, type SavedSong, type SongDetails } from './storage'

export type PanelView = 'generate' | 'export' | 'history'
const message = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong.'
const duration = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`

function Cover({ blob, title }: { blob: Blob | null; title: string }) {
  const image = useRef<HTMLImageElement>(null)
  useEffect(() => {
    if (!blob || !image.current) return
    const next = URL.createObjectURL(blob)
    image.current.src = next
    return () => URL.revokeObjectURL(next)
  }, [blob])
  return blob ? <img ref={image} alt={`${title} cover`} /> : <span className="cover-placeholder" aria-hidden="true">♫</span>
}

function ExportEditor({ song, working, saved, onExport }: { song: SavedSong; working: boolean; saved: boolean; onExport: (edits: ExportDetails) => Promise<boolean> }) {
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
    if (!busy) void onExport({ title, artist, cover, coverChanged }).then((saved) => { if (saved) setCoverChanged(false) })
  }}>
    <div className="panel-heading"><span>02 / EXPORT MAP</span><span className="export-ready">READY</span></div>
    <div className="export-cover-row">
      <label className={`export-cover${busy ? ' is-busy' : ''}`}>
        <Cover blob={cover} title={title || 'Song'} />
        <span>{coverBusy ? 'Loading…' : 'Change cover'}</span>
        <input type="file" accept="image/png,image/jpeg,image/webp" disabled={busy} aria-label="Album cover"
          onChange={(event) => { void changeCover(event.target.files?.[0]); event.target.value = '' }} />
      </label>
      <div className="export-cover-copy"><strong>Your map is ready.</strong>
        <p>{song.details.coverSource === 'generated' && !coverChanged ? 'No album art found. Add a cover, or keep this one.' : 'Review the song details before exporting.'}</p>
        <span>{duration(song.details.duration)} · {Math.round(song.details.bpm)} BPM</span>
      </div>
    </div>
    <label className="export-field">SONG NAME<input required maxLength={200} value={title} disabled={busy} onChange={(event) => setTitle(event.target.value)} /></label>
    <label className="export-field">ARTIST<input maxLength={200} value={artist} placeholder="Artist name" disabled={busy} onChange={(event) => setArtist(event.target.value)} /></label>
    <div className="export-charts">{song.details.charts.map((chart) => <span key={chart.difficulty}>{chart.difficulty === 'ExpertPlus' ? 'Expert+' : chart.difficulty} <small>{chart.notes.toLocaleString()} notes</small></span>)}</div>
    {error && <p className="error-message" role="alert">{error}</p>}
    <button type="submit" className="generate-button" disabled={busy || !title.trim()}><span>{working ? 'Preparing export…' : 'Save & download ZIP'}</span><span aria-hidden="true">↓</span></button>
    {!working && (dirty || saved) && <p className="library-save-status" role="status">{dirty ? 'Changes will be saved when you export' : 'Saved to your history'}</p>}
  </form>
}

type Props = { result: Result | null; view: PanelView; onViewChange: (view: PanelView) => void; onNew: () => void; generationBusy: boolean }

export default function MapLibrary({ result, view, onViewChange, onNew, generationBusy }: Props) {
  const [songs, setSongs] = useState<SongDetails[]>([])
  const [active, setActive] = useState<SavedSong | null>(null)
  const [working, setWorking] = useState(false)
  const [processedResult, setProcessedResult] = useState<Result | null>(null)
  const preparing = result !== null && processedResult !== result
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const [removeId, setRemoveId] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void listSongs().then((items) => { if (!cancelled) setSongs(items) })
      .catch((cause: unknown) => { if (!cancelled) setError(`History unavailable: ${message(cause)}`) })
      .finally(() => { if (!cancelled) setLoadingHistory(false) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (!result) return
    let cancelled = false
    void (async () => {
      try {
        const song = await songFromResult(result)
        if (cancelled) return
        setError(null)
        setSaved(false)
        setActive(song)
        try {
          await saveSong(song)
          if (cancelled) return
          setSaved(true)
          setSongs(await listSongs())
        } catch {
          if (!cancelled) setError('Could not save this map to history. You can still download it below.')
        }
      } catch (cause) {
        if (!cancelled) { setActive(null); setSaved(false); setError(`Could not open the export editor: ${message(cause)}`) }
      } finally { if (!cancelled) setProcessedResult(result) }
    })()
    return () => { cancelled = true }
  }, [result])

  const exportMap = async (edits: ExportDetails): Promise<boolean> => {
    if (!active || working) return false
    setWorking(true)
    setError(null)
    try {
      const updated = await updateSongExport(active, edits)
      setActive(updated)
      try { await saveSong(updated); setSongs(await listSongs()); setSaved(true) }
      catch { setSaved(false); setError('Download is ready, but these edits could not be saved to history. Browser storage may be full.') }
      downloadSong(updated)
      return true
    } catch (cause) { setError(message(cause)); return false }
    finally { setWorking(false) }
  }

  const openSong = async (id: string, downloadOnly = false) => {
    if (working || generationBusy) return
    setWorking(true)
    setError(null)
    try {
      const song = await getSong(id)
      if (downloadOnly) downloadSong(song)
      else { setActive(song); setSaved(true); onViewChange('export') }
    } catch (cause) { setError(message(cause)) }
    finally { setWorking(false) }
  }

  const removeSong = async (id: string) => {
    setWorking(true)
    setError(null)
    try { await deleteSong(id); setSongs(await listSongs()); setRemoveId(null); if (active?.details.id === id) setSaved(false) }
    catch (cause) { setError(message(cause)) }
    finally { setWorking(false) }
  }

  return <>
    <nav className="library-nav" aria-label="Map library">
      <button type="button" aria-current={view === 'generate' ? 'page' : undefined} disabled={generationBusy || working || preparing}
        onClick={() => { if (view !== 'generate') { setError(null); setActive(null); setProcessedResult(null); setSaved(false); onNew() } }}>New map</button>
      <button type="button" aria-current={view === 'history' ? 'page' : undefined} disabled={generationBusy || working || preparing}
        onClick={() => {
          setError(null); setLoadingHistory(true); onViewChange('history')
          void listSongs().then(setSongs).catch((cause: unknown) => setError(`History unavailable: ${message(cause)}`)).finally(() => setLoadingHistory(false))
        }}>History{songs.length > 0 && <span>{songs.length}</span>}</button>
    </nav>
    {view === 'export' && <>
      {preparing ? <p className="library-message" role="status">Preparing your export…</p>
        : active ? <ExportEditor key={active.details.id} song={active} working={working} saved={saved} onExport={exportMap} />
        : working ? <p className="library-message" role="status">Preparing your export…</p>
          : result && <button type="button" className="generate-button" onClick={() => {
            const link = document.createElement('a'); const url = URL.createObjectURL(result.zip)
            link.href = url; link.download = result.fileName; link.click(); setTimeout(() => URL.revokeObjectURL(url), 30_000)
          }}>Download original map</button>}
    </>}
    {view === 'history' && <section className="song-history" aria-label="Generated songs">
      <div className="panel-heading"><span>YOUR MAPS</span><span className="file-label">SAVED IN THIS BROWSER</span></div>
      {loadingHistory ? <p className="library-message">Loading history…</p> : songs.length === 0 ? <p className="library-message">Your generated maps will appear here.</p> : <ul>
        {songs.map((song) => <li key={song.id}>
          <div className="history-cover"><Cover blob={song.cover} title={song.title} /></div>
          <div className="history-song"><strong>{song.title}</strong><span>{song.artist || 'Unknown artist'}</span><small>{duration(song.duration)} · {new Date(song.createdAt).toLocaleDateString()}</small>
            <div className="history-actions">
              <button type="button" disabled={working} onClick={() => void openSong(song.id)}>Edit / export</button>
              <button type="button" disabled={working} onClick={() => void openSong(song.id, true)}>Download</button>
              <button type="button" disabled={working} className="history-remove" aria-label={`Remove ${song.title} from history`} onClick={() => setRemoveId(song.id)}>Remove</button>
            </div>
            {removeId === song.id && <div className="history-confirm"><span>Remove this saved map?</span><button type="button" disabled={working} onClick={() => void removeSong(song.id)}>Remove</button><button type="button" onClick={() => setRemoveId(null)}>Cancel</button></div>}
          </div>
        </li>)}
      </ul>}
    </section>}
    {error && view !== 'generate' && <p className="error-message" role="alert">{error}</p>}
  </>
}
