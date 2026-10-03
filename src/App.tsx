import { useEffect, useRef, useState } from 'react'
import { generate, type Result } from './engine/pipeline'
import { overallGenerationPercent } from './engine/progress'
import type { Difficulty, Palette, Progress, Settings } from './engine/types'
import { ENVIRONMENTS } from './engine/style'
import Backdrop from './scene/Backdrop'
import MapLibrary, { type PanelView } from './library/MapLibrary'
import { clearSongHistory } from './library/storage'
import { DEV_TOOLS } from './dev/flags'
import { deletePreloadedModels, resetPreloadForDevelopment, startPreload, subscribePreload, type PreloadState } from './engine/preload'

const stages: Record<Progress['stage'], string> = {
  decode: 'Decoding audio',
  stems: 'Separating stems',
  beats: 'Tracking beats',
  vocals: 'Finding vocals',
  attacks: 'Detecting attacks',
  grid: 'Building beat grid',
  sections: 'Finding song sections',
  tokens: 'Building song tokens',
  candidates: 'Placing candidates',
  notes: 'Writing notes',
  walls: 'Planning walls',
  lights: 'Lighting',
  style: 'Styling the map',
  package: 'Packaging',
}

const DIFFICULTIES: [Difficulty, string][] = [['Expert', 'Expert'], ['ExpertPlus', 'Expert+']]
const WALLS: [Settings['walls'], string][] = [['off', 'Off'], ['light', 'Light'], ['normal', 'Normal'], ['heavy', 'Heavy']]
const LIGHTING: [Settings['lighting'], string][] = [['calm', 'Calm'], ['normal', 'Normal'], ['intense', 'Intense']]
const CUSTOM_COLORS: Palette = { left: '#e61940', right: '#2fa8ff', lightA: '#ff3152', lightB: '#42d7ff' }
const COLOR_LABELS: [keyof Palette, string][] = [['left', 'Left'], ['right', 'Right'], ['lightA', 'Light A'], ['lightB', 'Light B']]

const randomSeed = () => Math.floor(Math.random() * 1e9)
const envName = (name: string) => name.replace(/Environment$/, '').replace(/([a-z])([A-Z])/g, '$1 $2')
const initialSettings = (): Settings => ({
  difficulties: ['Expert', 'ExpertPlus'],
  walls: 'auto',
  arcs: true,
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
  const [file, setFile] = useState<File | null>(null)
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

  // Dev tools: Ctrl+X resets models and history, Ctrl+S runs a fake generation.
  const fakeRun = useRef<() => void>(() => {})
  useEffect(() => {
    if (!DEV_TOOLS) return
    const onKey = async (event: KeyboardEvent) => {
      const key = event.key.toLowerCase()
      if (!(event.ctrlKey || event.metaKey) || (key !== 'x' && key !== 's') || event.repeat) return
      const target = event.target as HTMLElement | null
      if (target?.isContentEditable || target?.closest('input, textarea, select')) return
      if (key === 's') {
        event.preventDefault()
        fakeRun.current()
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

  const selectFile = (next: File | null) => {
    if (busy) return
    setFile(next)
    setResult(null)
    setProgress(null)
    setError(null)
  }
  const abort = useRef<AbortController | null>(null)
  const busy = progress !== null && result === null && error === null
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

  const run = async (source: File, generator: typeof generate) => {
    setResult(null)
    setError(null)
    setOverallProgress(0)
    abort.current = new AbortController()
    try {
      setResult(await generator(source, { ...settings }, (next) => {
        setProgress(next)
        setOverallProgress((current) => Math.max(current, overallGenerationPercent(next)))
      }, abort.current.signal))
      setPanelView('export')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }
  const start = () => { if (file && !busy) void run(file, generate) }
  useEffect(() => {
    // import.meta.env.DEV inline lets the production build drop the fake chunk entirely.
    if (import.meta.env.DEV && DEV_TOOLS) fakeRun.current = () => {
      if (busy || !settings.difficulties.length) return
      void import('./dev/fake-generate').then(({ fakeGenerate, fakeSongFile }) => {
        const source = fakeSongFile()
        setPanelView('generate')
        setFile(source)
        setProgress(null)
        return run(source, fakeGenerate)
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
              <MapLibrary result={result} view={panelView} onViewChange={setPanelView} generationBusy={busy}
                onNew={() => { setPanelView('generate'); setResult(null); setProgress(null); setError(null); setFile(null); setSettings(initialSettings()) }} />
              {panelView === 'generate' && <>
              <div className="panel-heading"><span>01 / SELECT TRACK</span><span className="file-label">AUDIO INPUT</span></div>
              <label
                className={`upload-zone${dragging ? ' is-dragging' : ''}${file ? ' has-file' : ''}${busy ? ' is-busy' : ''}`}
                onDragOver={(event) => { event.preventDefault(); if (!busy) setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault()
                  setDragging(false)
                  if (busy) return
                  const next = event.dataTransfer.files[0]
                  if (next && (next.type.startsWith('audio/') || /\.(mp3|m4a|ogg|wav|flac)$/i.test(next.name))) selectFile(next)
                  else setError('Choose an MP3, M4A, OGG, WAV or FLAC audio file.')
                }}
              >
                <input id="audio" type="file" accept="audio/*,.mp3,.m4a,.ogg,.wav,.flac" disabled={busy}
                  onChange={(event) => selectFile(event.target.files?.[0] ?? null)} />
                <span className="upload-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4">
                    {file ? <><path d="M9 18V5l11-2v13M9 9l11-2" /><ellipse cx="6" cy="18" rx="3" ry="2" /><ellipse cx="17" cy="16" rx="3" ry="2" /></> : <><path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5" /></>}
                  </svg>
                </span>
                <span className="upload-title">{file ? file.name : 'Drop your track here'}</span>
                <span className="upload-description">{file ? 'Click to choose a different track' : <>or <span>browse files</span> to get started</>}</span>
                <span className="audio-formats">MP3 · M4A · OGG · WAV · FLAC</span>
              </label>
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
                  <label className="advanced-field"><span>WALLS</span>
                    <select value={settings.walls} onChange={(e) => update({ walls: e.target.value as Settings['walls'] })}>
                      <option value="auto">{auto(detected && (detected.walls.enabled ? `${detected.walls.perMinute.toFixed(1)} / min` : 'None'))}</option>
                      {WALLS.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
                    </select>
                  </label>
                  <label className="advanced-field"><span>ARCS</span>
                    <select value={settings.arcs ? 'auto' : 'off'} onChange={(e) => update({ arcs: e.target.value === 'auto' })}>
                      <option value="auto">{auto(detected && `${detected.charts.reduce((n, c) => n + c.arcs, 0)} arcs`)}</option>
                      <option value="off">Off</option>
                    </select>
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
                    <input type="text" value={settings.title} placeholder={auto(detected?.title ?? 'from tags')} onChange={(e) => update({ title: e.target.value })} />
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
                  <label className="advanced-field"><span>CANDIDATES</span>
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
              <button id="generate" type="button" disabled={!file || busy || !settings.difficulties.length} onClick={start} className="generate-button">
                <span>{busy ? 'Creating your map…' : 'Generate map'}</span>
                {busy ? <span className="spinner" aria-hidden="true" /> : <span aria-hidden="true">↗</span>}
              </button>

              {progress && (
                <section className="progress-section" aria-live="polite">
                  <div className="progress-label"><span>{stages[progress.stage]}</span><span>{Math.round(progress.fraction * 100)}%</span></div>
                  <div className="progress-track" role="progressbar" aria-label={stages[progress.stage]} aria-valuenow={Math.round(progress.fraction * 100)} aria-valuemin={0} aria-valuemax={100}>
                    <div style={{ width: `${progress.fraction * 100}%` }} />
                  </div>
                  {progress.detail && <p>{progress.detail}</p>}
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
