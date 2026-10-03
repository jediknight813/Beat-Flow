import { selectBackend, type BackendPreference } from './backend'
import type { Progress, Settings } from './types'
import ortMjs from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url'
import ortWasm from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url'
import { configure, loadModel, ort, releaseGroup, type Backend, type ModelManifest } from './models'
import { runPipeline, type CoreResult, type Deps, type StereoAudio } from './core'
import { cachedFetch } from './model-cache'
import type { StrainModel } from './flow'
import { coverPixels, loadEnvironmentReference } from './style'

export type WorkerRequest = {
  type: 'generate'
  backendPreference: BackendPreference
  id: number
  audio: StereoAudio
  file: { name: string; bytes: ArrayBuffer | null }
  settings: Settings
}

export type WorkerResponse =
  | { type: 'progress'; id: number; progress: Progress }
  | { type: 'done'; id: number; result: CoreResult }
  | { type: 'error'; id: number; message: string }

type Scope = {
  postMessage(message: WorkerResponse): void
  addEventListener(type: 'message', listener: (event: MessageEvent<WorkerRequest>) => void): void
}

const scope = self as unknown as Scope

const LABELS: Record<string, string> = {
  demucs: 'Stem separation (Demucs)',
  beatthis: 'Beat tracking (Beat This!)',
  crepe: 'Vocal pitch (CREPE)',
  lights: 'Lighting',
}

async function json<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url)
    return res.ok ? ((await res.json()) as T) : null
  } catch {
    return null
  }
}

let backend: Promise<Backend> | null = null

function ready(preference: BackendPreference): Promise<Backend> {
  backend ??= selectBackend(preference).then((selection) => {
    // This installed ORT JSEP backend takes an adapter and requests all required
    // device limits/features itself. Bind the chosen adapter before any session.
    if (selection.adapter) ort.env.webgpu.adapter = selection.adapter as typeof ort.env.webgpu.adapter
    configure(selection.backend, { mjs: ortMjs, wasm: ortWasm })
    return selection.backend
  })
  return backend
}

async function deps(preference: BackendPreference): Promise<Deps> {
  const chosen = await ready(preference)
  return {
    backend: chosen,
    async model(group, file, onProgress) {
      const base = `${import.meta.env.BASE_URL}models/${group}/`
      const label = LABELS[group] ?? `${group} model`
      let name = file
      if (!name) {
        const spec = await json<Partial<Record<Backend, string | string[]>>>(`${base}spec.json`)
        const pick = spec?.[chosen]
        name = Array.isArray(pick) ? pick[0] : pick
      }
      const manifests = await json<Record<string, ModelManifest>>(`${base}manifest.json`)
      if (!name || !manifests?.[name]) throw new Error(`${label} model not available yet`)
      return loadModel(group, name, group === 'flow' ? 'wasm' : chosen, onProgress)
    },
    release: releaseGroup,
    async strainData() {
      const response = await cachedFetch(`${import.meta.env.BASE_URL}models/flow/strain.json`, 'flow-1')
      return await response.json() as StrainModel
    },
    environmentReference: loadEnvironmentReference,
    async coverPixels(cover) {
      try {
        const bitmap = await createImageBitmap(cover)
        const pixels = coverPixels(bitmap)
        bitmap.close()
        return pixels
      } catch {
        return null
      }
    },
  }
}

scope.addEventListener('message', async (event) => {
  const { id, audio, file, settings, backendPreference } = event.data
  try {
    const result = await runPipeline(
      audio,
      { name: file.name, bytes: file.bytes ? new Uint8Array(file.bytes) : null },
      settings,
      await deps(backendPreference),
      (progress) => scope.postMessage({ type: 'progress', id, progress }),
    )
    scope.postMessage({ type: 'done', id, result })
  } catch (e) {
    scope.postMessage({ type: 'error', id, message: e instanceof Error ? e.message : String(e) })
  }
})
