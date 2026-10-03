import { useEffect, useRef, useState } from 'react'
import { generate, type Result } from './engine/pipeline'
import { generationPhases, generationStages, overallGenerationPercent } from './engine/progress'
import type { Difficulty, Palette, Progress, Settings } from './engine/types'
import { ENVIRONMENTS } from './engine/style'
import Backdrop from './scene/Backdrop'
import MapLibrary, { type PanelView } from './library/MapLibrary'
import { clearSongHistory, listSongs, saveSong } from './library/storage'
import { songFromResult } from './library/archive'
import { DEV_TOOLS } from './dev/flags'
import { deletePreloadedModels, resetPreloadForDevelopment, startPreload, subscribePreload, type PreloadState } from './engine/preload'

const DIFFICULTIES: [Difficulty, string][] = [['Expert', 'Expert'], ['ExpertPlus', 'Expert+']]
const LIGHTING: [Settings['lighting'], string][] = [['calm', 'Calm'], ['normal', 'Normal'], ['intense', 'Intense']]
const CUSTOM_COLORS: Palette = { left: '#e61940', right: '#2fa8ff', lightA: '#ff3152', lightB: '#42d7ff' }
const COLOR_LABELS: [keyof Palette, string][] = [['left', 'Left'], ['right', 'Right'], ['lightA', 'Light A'], ['lightB', 'Light B']]

type TrackStatus = 'pending' | 'running' | 'done' | 'failed'
const TRACK_STATUS: Record<TrackStatus, string> = { pending: 'Queued', running: 'Generating', done: 'Saved', failed: 'Failed' }
const isAudio = (file: File) => file.type.startsWith('audio/') || /\.(mp3|m4a|ogg|wav|flac)$/i.test(file.name)
const errorText = (cause: unknown) => cause instanceof Error ? cause.message : String(cause)

const randomSeed = () => Math.floor(Math.random() * 1e9)
const envName = (name: string) => name.replace(/Environment$/, '').replace(/([a-z])([A-Z])/g, '$1 $2')
const initialSettings = (): Settings => ({
  difficulties: ['Expert', 'ExpertPlus'],
  walls: 'off',
  arcs: false,
  lighting: 'auto',
  environment: 'auto',
  colors: 'auto',
  title: '',
  artist: '',
  cover: 'auto',
  seed: randomSeed(),
  candidates: 4,
})

export default function App() {
  const [settings, setSettings] = useState<Settings>(initialSettings)
  const update = (next: Partial<Settings>) => setSettings((current) => ({ ...current, ...next }))
  const [files, setFiles] = useState<File[]>([])
  const [tracks, setTracks] = useState<TrackStatus[]>([])
  const [trackIndex, setTrackIndex] = useState(0)
  const [running, setRunning] = useState(false)
  const [historyCount, setHistoryCount] = useState(0)
  const [progress, setProgress] = useState<Progress | null>(null)
  const [overallProgress, setOverallProgress] = useState(0)
  const [result, setResult] = useState<Result | null>(null)
  const [panelView, setPanelView] = useState<PanelView>('generate')
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [models, setModels] = useState<PreloadState | null>(null)
  const [modelMenuOpen, setModelMenuOpen] = useState(false)
  const [deletingModels, setDeletingModels] = useState(false)
  const [modelActionError, setModelActionError] = useState<string | null>(null)
  const modelButton = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    const unsubscribe = subscribePreload(setModels)
    void startPreload()
    return unsubscribe
  }, [])

  // Dev tools: Ctrl+X resets models and history, Ctrl+S runs a fake generation
  // (Ctrl+Shift+S fakes a three-track album).
  const fakeRun = useRef<(album: boolean) => void>(() => {})
  useEffect(() => {
    if (!DEV_TOOLS) return
    const onKey = async (event: KeyboardEvent) => {
      const key = event.key.toLowerCase()
      if (!(event.ctrlKey || event.metaKey) || (key !== 'x' && key !== 's') || event.repeat) return
      const target = event.target as HTMLElement | null
      if (target?.isContentEditable || target?.closest('input, textarea, select')) return
      if (key === 's') {
        event.preventDefault()
        fakeRun.current(event.shiftKey)
        return
      }
      if (window.getSelection()?.toString()) return
      event.preventDefault()
      try {
        await Promise.all([resetPreloadForDevelopment(), clearSongHistory()])
        window.location.reload()
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : 'Unable to delete downloaded models.')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const modelLabel = deletingModels ? 'Deleting models…' : models?.deleted ? 'Models deleted' : !models || (!models.total && !models.done && !models.error)
    ? 'Runs in your browser'
    : models.error
      ? 'Models unavailable'
      : models.done
        ? `Model downloaded · ${models.backend === 'webgpu' ? 'WebGPU' : 'CPU'}`
        : `Loading models ${Math.round((models.loaded / Math.max(1, models.total)) * 100)}%`

  const selectFiles = (picked: File[]) => {
    if (busy) return
    const next = picked.filter(isAudio)
    setFiles(next)
    setTracks(next.map(() => 'pending'))
    setResult(null)
    setProgress(null)
    const skipped = picked.length - next.length
    setError(skipped ? `Skipped ${skipped} non-audio file${skipped === 1 ? '' : 's'}.` : null)
  }
  const removeFile = (index: number) => {
    if (busy) return
    setFiles((current) => current.filter((_, i) => i !== index))
    setTracks((current) => current.filter((_, i) => i !== index))
  }
  const abort = useRef<AbortController | null>(null)
  const busy = running
  // Generating mid-download would fetch the same models a second time in the worker.
  const modelsDownloading = !models || (!models.done && !models.error && !models.deleted)
  const modelPercent = models?.total ? Math.round((models.loaded / models.total) * 100) : 0
  const album = files.length > 1
  useEffect(() => {
    document.title = busy ? `${overallProgress}% · BeatFlow` : 'BeatFlow'
    return () => { document.title = 'BeatFlow' }
  }, [busy, overallProgress])
  const detected = result?.detected
  const auto = (value?: string) => (value ? `Auto · ${value}` : 'Auto')
  const palette = settings.colors === 'auto' ? detected?.palette : settings.colors
  const toggleDifficulty = (d: Difficulty) => update({
    difficulties: settings.difficulties.includes(d) ? settings.difficulties.filter((x) => x !== d) : DIFFICULTIES.map(([x]) => x).filter((x) => x === d || settings.difficulties.includes(x)),
  })

  const deleteModels = async () => {
    if (deletingModels || busy) return
    setDeletingModels(true)
    setModelActionError(null)
    try {
      await deletePreloadedModels()
    } catch (cause) {
      setModelActionError(cause instanceof Error ? cause.message : 'Unable to delete downloaded models.')
    } finally {
      setDeletingModels(false)
    }
  }

  // One track opens the export editor. An album generates each track in turn,
  // saves it straight to history and keeps going if a single track fails.
  const run = async (sources: File[], generator: typeof generate) => {
    const batch = sources.length > 1
    const mark = (index: number, status: TrackStatus) => setTracks((current) => current.map((s, i) => (i === index ? status : s)))
    setRunning(true)
    setResult(null)
    setError(null)
    setProgress(null)
    setOverallProgress(0)
    setTracks(sources.map(() => 'pending'))
    const controller = abort.current = new AbortController()
    let failed = 0
    try {
      for (const [index, source] of sources.entries()) {
        setTrackIndex(index)
        mark(index, 'running')
        try {
          const next = await generator(source, batch ? { ...settings, title: '' } : { ...settings }, (p) => {
            setProgress(p)
            setOverallProgress((current) => Math.max(current, Math.floor((index * 100 + overallGenerationPercent(p)) / sources.length)))
          }, controller.signal)
          if (!batch) {
            setResult(next)
            setPanelView('export')
            return
          }
          await saveSong(await songFromResult(next))
          setHistoryCount((await listSongs()).length)
          mark(index, 'done')
        } catch (cause) {
          if (!batch || (cause instanceof DOMException && cause.name === 'AbortError')) throw cause
          failed++
          mark(index, 'failed')
        }
      }
      setProgress(null)
      if (failed) setError(`${failed} of ${sources.length} tracks failed. The rest are saved to your history.`)
      else setPanelView('history')
    } catch (cause) {
      setError(errorText(cause))
    } finally {
      setRunning(false)
    }
  }
  const start = () => { if (files.length && !busy && !modelsDownloading) void run(files, generate) }
  const openNew = () => {
    if (panelView === 'generate') return
    setPanelView('generate')
    if (busy) return
    setResult(null); setProgress(null); setError(null); setFiles([]); setTracks([]); setSettings(initialSettings())
  }
  useEffect(() => {
    // import.meta.env.DEV inline lets the production build drop the fake chunk entirely.
    if (import.meta.env.DEV && DEV_TOOLS) fakeRun.current = (fakeAlbum) => {
      if (busy || !settings.difficulties.length) return
      void import('./dev/fake-generate').then(({ fakeGenerate, fakeSongFiles }) => {
        const sources = fakeSongFiles(fakeAlbum ? 3 : 1)
        setPanelView('generate')
        setFiles(sources)
        return run(sources, fakeGenerate)
      })
    }
  })

  return (
    <>
      <Backdrop />
      <div className="scene-shade" aria-hidden="true" />
      <div className="app-shell">
        <header className="masthead">
          <a className="wordmark" href="./" aria-label="BeatFlow home">
            <span className="brand-icon" aria-hidden="true"><i /><i /></span>
            BEAT<span>FLOW</span>
          </a>
          {historyCount > 0 && <nav className="site-nav" aria-label="Main">
            <button type="button" aria-current={panelView !== 'history' ? 'page' : undefined} onClick={openNew}>New map</button>
            <button type="button" aria-current={panelView === 'history' ? 'page' : undefined} onClick={() => setPanelView('history')}>History<span>{historyCount}</span></button>
          </nav>}
          <div className="model-status"
            onMouseEnter={() => setModelMenuOpen(true)}
            onMouseLeave={() => { if (!deletingModels) setModelMenuOpen(false) }}
            onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setModelMenuOpen(false) }}
            onKeyDown={(event) => { if (event.key === 'Escape') { setModelMenuOpen(false); modelButton.current?.focus() } }}>
            <button ref={modelButton} type="button" className="local-badge" aria-expanded={modelMenuOpen} aria-controls="model-actions"
              onClick={() => setModelMenuOpen((open) => !open)}><i /> {modelLabel}</button>
            {modelMenuOpen && <div className="model-popover" id="model-actions">
              {models?.deleted || models?.error ? <button type="button" onClick={() => { setModelActionError(null); void startPreload(true) }}>Download models</button>
                : models?.done ? <button type="button" className="delete-models" disabled={deletingModels || busy} onClick={() => void deleteModels()}>
                  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.2" aria-hidden="true"><path d="M3 4.5h10M6 4.5V2.8h4v1.7M4.5 4.5l.6 8.5h5.8l.6-8.5M6.5 6.5v4M9.5 6.5v4" /></svg>
                  <span>{deletingModels ? 'Deleting…' : 'Delete models'}</span>
                </button>
                  : <p>Downloading models…</p>}
              {busy && <p>Available when generation finishes.</p>}
              {(modelActionError || models?.error) && <p role="alert">{modelActionError || models?.error}</p>}
            </div>}
          </div>
        </header>

        <main className="main-content">
          <div className="studio">
            <div className="intro">
              <h1>Your song.<br /><span>Your stage.</span></h1>
            </div>

            <section className="generator" aria-label={panelView === 'export' ? 'Export your map' : panelView === 'history' ? 'Song history' : 'Create a Beat Saber map'}>
              <MapLibrary result={result} view={panelView} onViewChange={setPanelView} onHistoryChange={setHistoryCount} generationBusy={busy} />
              {panelView === 'generate' && <>
              <div className="panel-heading"><span>01 / SELECT {album ? 'TRACKS' : 'TRACK'}</span><span className="file-label">AUDIO INPUT</span></div>
              <label
                className={`upload-zone${dragging ? ' is-dragging' : ''}${files.length ? ' has-file' : ''}${busy ? ' is-busy' : ''}`}
                onDragOver={(event) => { event.preventDefault(); if (!busy) setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault()
                  setDragging(false)
                  if (busy) return
                  const picked = [...event.dataTransfer.files]
                  if (picked.some(isAudio)) selectFiles(picked)
                  else setError('Choose MP3, M4A, OGG, WAV or FLAC audio files.')
                }}
              >
                <input id="audio" type="file" multiple accept="audio/*,.mp3,.m4a,.ogg,.wav,.flac" disabled={busy}
                  onChange={(event) => { selectFiles([...(event.target.files ?? [])]); event.target.value = '' }} />
                <span className="upload-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4">
                    {files.length ? <><path d="M9 18V5l11-2v13M9 9l11-2" /><ellipse cx="6" cy="18" rx="3" ry="2" /><ellipse cx="17" cy="16" rx="3" ry="2" /></> : <><path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5" /></>}
                  </svg>
                </span>
                <span className="upload-title">{album ? `${files.length} tracks selected` : files.length ? files[0].name : 'Drop your tracks here'}</span>
                <span className="upload-description">{album ? 'Click to choose different tracks' : files.length ? 'Click to choose a different track, or add more for an album' : <>One song or a whole album · <span>browse files</span></>}</span>
                <span className="audio-formats">MP3 · M4A · OGG · WAV · FLAC</span>
              </label>
              {album && <ol className="track-list" aria-label="Tracks to generate">
                {files.map((track, index) => <li key={`${track.name}-${index}`} className={`is-${tracks[index] ?? 'pending'}`}>
                  <span className="track-number">{String(index + 1).padStart(2, '0')}</span>
                  <span className="track-name">{track.name.replace(/\.[^.]+$/, '')}</span>
                  {busy || tracks[index] === 'done' || tracks[index] === 'failed'
                    ? <span className="track-status">{TRACK_STATUS[tracks[index] ?? 'pending']}</span>
                    : <button type="button" aria-label={`Remove ${track.name}`} onClick={() => removeFile(index)}>×</button>}
                </li>)}
              </ol>}
              <details className="advanced">
                <summary><span>ADVANCED</span><span className="file-label">{detected ? 'DETECTED' : 'AUTO'}</span></summary>
                <fieldset disabled={busy}>
                  <div className="advanced-field"><span>DIFFICULTIES</span>
                    <div className="advanced-chips">
                      {DIFFICULTIES.map(([d, label]) => (
                        <button key={d} type="button" aria-pressed={settings.difficulties.includes(d)} onClick={() => toggleDifficulty(d)}>{label}</button>
                      ))}
                    </div>
                  </div>
                  <label className="advanced-field"><span>TEMPO</span>
                    <output>{auto(detected && `${Math.round(detected.bpm)} BPM · ${detected.sections} sections`)}</output>
                  </label>
                  <label className="advanced-field"><span>LIGHTING</span>
                    <select value={settings.lighting} onChange={(e) => update({ lighting: e.target.value as Settings['lighting'] })}>
                      <option value="auto">{auto(detected && LIGHTING.find(([v]) => v === detected.lighting)?.[1])}</option>
                      {LIGHTING.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
                    </select>
                  </label>
                  <label className="advanced-field"><span>ENVIRONMENT</span>
                    <select value={settings.environment} onChange={(e) => update({ environment: e.target.value })}>
                      <option value="auto">{auto(detected && envName(detected.environment))}</option>
                      {ENVIRONMENTS.map((env) => <option key={env} value={env}>{envName(env)}</option>)}
                    </select>
                  </label>
                  <div className="advanced-field"><span>COLOURS</span>
                    <div className="advanced-chips">
                      <select value={settings.colors === 'auto' ? 'auto' : 'custom'} onChange={(e) => update({ colors: e.target.value === 'auto' ? 'auto' : detected?.palette ?? CUSTOM_COLORS })}>
                        <option value="auto">Auto</option>
                        <option value="custom">Custom</option>
                      </select>
                      {palette && COLOR_LABELS.map(([key, label]) => (
                        <input key={key} type="color" className="advanced-swatch" title={label} aria-label={`${label} colour`} value={palette[key]}
                          onChange={(e) => update({ colors: { ...palette, [key]: e.target.value } })} />
                      ))}
                    </div>
                  </div>
                  <label className="advanced-field"><span>TITLE</span>
                    <input type="text" value={album ? '' : settings.title} disabled={album} placeholder={album ? "Auto · each track's tags" : auto(detected?.title ?? 'from tags')} onChange={(e) => update({ title: e.target.value })} />
                  </label>
                  <label className="advanced-field"><span>ARTIST</span>
                    <input type="text" value={settings.artist} placeholder={auto(detected?.artist || 'from tags')} onChange={(e) => update({ artist: e.target.value })} />
                  </label>
                  <div className="advanced-field"><span>COVER</span>
                    <div className="advanced-chips">
                      <label className="advanced-file">
                        <input type="file" accept="image/png,image/jpeg" onChange={(e) => { const f = e.target.files?.[0]; if (f) update({ cover: f }) }} />
                        {settings.cover === 'auto' ? auto(detected && (detected.cover ? 'Embedded art' : 'Generated')) : 'Custom image'}
                      </label>
                      {settings.cover !== 'auto' && <button type="button" onClick={() => update({ cover: 'auto' })}>Auto</button>}
                    </div>
                  </div>
                  <label className="advanced-field"><span>VARIATIONS</span>
                    <select value={settings.candidates} onChange={(e) => update({ candidates: Number(e.target.value) })}>
                      {[1, 2, 3, 4, 6, 8].map((n) => <option key={n} value={n}>{n === 4 ? 'Auto · 4' : n}</option>)}
                    </select>
                  </label>
                  <div className="advanced-field"><span>SEED</span>
                    <div className="advanced-chips">
                      <input type="number" value={settings.seed} onChange={(e) => update({ seed: Math.max(0, Math.floor(Number(e.target.value) || 0)) })} />
                      <button type="button" aria-label="New seed" onClick={() => update({ seed: randomSeed() })}>↻</button>
                    </div>
                  </div>
                </fieldset>
              </details>
              <button id="generate" type="button" disabled={!files.length || busy || modelsDownloading || !settings.difficulties.length} onClick={start} className="generate-button">
                <span>{busy ? (album ? `Creating map ${trackIndex + 1} of ${files.length}…` : 'Creating your map…') : modelsDownloading && files.length ? `Downloading models… ${modelPercent}%` : album ? `Generate ${files.length} maps` : 'Generate map'}</span>
                {busy ? <span className="spinner" aria-hidden="true" /> : <span aria-hidden="true">↗</span>}
              </button>

              {progress && busy && (
                <section className="progress-section" aria-label="Map generation">
                  <div className="progress-heading">
                    <div>
                      <span className="progress-eyebrow">{album ? `TRACK ${trackIndex + 1} OF ${files.length}` : 'CREATING YOUR MAP'}</span>
                      <h2 aria-live="polite">{generationStages[progress.stage].label}</h2>
                    </div>
                    <span className="progress-percent">{overallProgress}<small>%</small></span>
                  </div>
                  <div className="progress-track" role="progressbar" aria-label={album ? 'Overall album progress' : 'Overall map progress'} aria-valuenow={overallProgress} aria-valuemin={0} aria-valuemax={100} aria-valuetext={`${overallProgress}% overall · ${generationStages[progress.stage].label}`}>
                    <div style={{ width: `${overallProgress}%` }} />
                  </div>
                  <ol className="progress-phases" aria-label="Generation steps">
                    {generationPhases.map((phase, index) => <li key={phase} className={index < generationStages[progress.stage].phase ? 'is-complete' : index === generationStages[progress.stage].phase ? 'is-current' : ''} aria-current={index === generationStages[progress.stage].phase ? 'step' : undefined}>
                      <span aria-hidden="true">{index < generationStages[progress.stage].phase ? '✓' : String(index + 1).padStart(2, '0')}</span>{phase}
                    </li>)}
                  </ol>
                  <p className="progress-detail" aria-live="polite">{progress.detail || generationStages[progress.stage].detail}</p>
                </section>
              )}
              {error && <p className="error-message" role="alert">{error}</p>}
              </>}
            </section>
          </div>
        </main>
      </div>
    </>
  )
}
