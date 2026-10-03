import { cacheModelFile, clearModelCache, type ModelManifest } from './model-cache'
import { pickBackend, type Backend } from './backend'

const GROUPS: Record<Backend, string[]> = {
  webgpu: ['demucs', 'beatthis', 'crepe', 'flow-1-v8-notes-fp16', 'flow', 'lights'],
  wasm: ['demucs', 'beatthis', 'crepe', 'flow-1-v8-notes-int8', 'flow', 'lights'],
}

export type PreloadState = {
  backend: Backend | null
  loaded: number
  total: number
  done: boolean
  persisted: boolean
  error: string | null
  deleted: boolean
}

type Spec = { webgpu?: string | string[]; wasm?: string | string[] }

const emptyState: PreloadState = { backend: null, loaded: 0, total: 0, done: false, persisted: false, error: null, deleted: false }
let state: PreloadState = { ...emptyState }
const DISABLED_KEY = 'beatflow-model-preload-disabled'
const listeners = new Set<(s: PreloadState) => void>()
let started: Promise<void> | null = null
let downloadController: AbortController | null = null

function emit(next: Partial<PreloadState>) {
  state = { ...state, ...next }
  for (const l of listeners) l(state)
}

export function subscribePreload(listener: (s: PreloadState) => void): () => void {
  listeners.add(listener)
  listener(state)
  return () => listeners.delete(listener)
}

export function preloadState(): PreloadState {
  return state
}

async function persist(): Promise<boolean> {
  try {
    if (!navigator.storage?.persist) return false
    if (await navigator.storage.persisted()) return true
    return await navigator.storage.persist()
  } catch {
    return false
  }
}

async function json<T>(url: string, signal?: AbortSignal): Promise<T | null> {
  try {
    const res = await fetch(url, { signal })
    return res.ok ? ((await res.json()) as T) : null
  } catch {
    return null
  }
}

async function plan(backend: Backend, signal: AbortSignal): Promise<{ group: string; manifest: ModelManifest }[]> {
  const out: { group: string; manifest: ModelManifest }[] = []
  for (const group of GROUPS[backend]) {
    signal.throwIfAborted()
    const base = `${import.meta.env.BASE_URL}models/${group}/`
    const manifests = await json<Record<string, ModelManifest>>(`${base}manifest.json`, signal)
    if (!manifests) continue
    const spec = await json<Spec>(`${base}spec.json`, signal)
    const pick = spec?.[backend]
    const names = pick ? (Array.isArray(pick) ? pick : [pick]) : Object.keys(manifests)
    for (const name of names) if (manifests[name]) out.push({ group, manifest: manifests[name] })
  }
  return out
}

export async function deletePreloadedModels(): Promise<void> {
  // Finish pending cache writes before deleting so they cannot recreate it.
  downloadController?.abort()
  await started
  await clearModelCache()
  try { localStorage.setItem(DISABLED_KEY, 'true') } catch { /* Session state still records the deletion. */ }
  started = null
  emit({ ...emptyState, deleted: true })
}

// Keep downloaded assets, but cancel the old plan before starting the new one.
export async function restartPreload(): Promise<void> {
  downloadController?.abort()
  await started
  started = null
  emit({ ...emptyState, deleted: state.deleted })
  void startPreload()
}

// The developer shortcut simulates a new visitor, including automatic downloads.
export async function resetPreloadForDevelopment(): Promise<void> {
  downloadController?.abort()
  await started
  await clearModelCache()
  try { localStorage.removeItem(DISABLED_KEY) } catch { /* Storage is unavailable; session state is reset below. */ }
  started = null
  emit(emptyState)
}

export function startPreload(force = false): Promise<void> {
  if (force && (state.deleted || state.error)) {
    started = null
    emit(emptyState)
    try { localStorage.removeItem(DISABLED_KEY) } catch { /* Downloads can still run without localStorage. */ }
  }
  let disabled = state.deleted
  try { disabled ||= localStorage.getItem(DISABLED_KEY) === 'true' } catch { /* Use session state. */ }
  if (disabled && !force) {
    emit({ ...emptyState, deleted: true })
    return Promise.resolve()
  }
  started ??= (async () => {
    downloadController = new AbortController()
    const { signal } = downloadController
    try {
      const [backend, persisted] = await Promise.all([pickBackend(), persist()])
      const files = await plan(backend, signal)
      signal.throwIfAborted()
      emit({ backend, persisted, total: files.reduce((n, f) => n + f.manifest.bytes, 0) })
      let loaded = 0
      for (const { group, manifest } of files) {
        await cacheModelFile(group, manifest, (bytes) => emit({ loaded: loaded + bytes }), signal)
        loaded += manifest.bytes
        emit({ loaded })
      }
      emit({ done: true })
    } catch (e) {
      if (signal.aborted) return
      emit({ error: e instanceof Error ? e.message : String(e) })
    }
  })()
  return started
}
