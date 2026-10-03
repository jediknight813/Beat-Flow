import * as ort from 'onnxruntime-web'
import { cachedFetch, type ModelManifest } from './model-cache'
import type { Backend } from './backend'

export { pickBackend, type Backend } from './backend'
export { cacheModelFile, cachedModels, clearModelCache, type ModelManifest } from './model-cache'

const sessions = new Map<string, Promise<ort.InferenceSession>>()

export function configure(backend: Backend, paths?: { mjs: string; wasm: string }) {
  if (paths) ort.env.wasm.wasmPaths = paths
  ort.env.wasm.numThreads = backend === 'wasm' && globalThis.crossOriginIsolated ? Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 2) - 1)) : 1
}

async function fetchChunks(base: string, manifest: ModelManifest, onProgress?: (fraction: number) => void): Promise<Uint8Array> {
  const out = new Uint8Array(manifest.bytes)
  let offset = 0
  for (const [i, file] of manifest.files.entries()) {
    const res = await cachedFetch(`${base}${file}`, String(manifest.bytes))
    const buf = new Uint8Array(await res.arrayBuffer())
    out.set(buf, offset)
    offset += buf.length
    onProgress?.((i + 1) / manifest.files.length)
  }
  return out.subarray(0, offset)
}

export function loadModel(group: string, file: string, backend: Backend, onProgress?: (fraction: number) => void): Promise<ort.InferenceSession> {
  const key = `${group}/${file}:${backend}`
  const existing = sessions.get(key)
  if (existing) return existing
  const base = `${import.meta.env.BASE_URL}models/${group}/`
  const promise = (async () => {
    const manifests = (await (await fetch(`${base}manifest.json`)).json()) as Record<string, ModelManifest>
    const manifest = manifests[file]
    if (!manifest) throw new Error(`unknown model: ${group}/${file}`)
    const bytes = await fetchChunks(base, manifest, onProgress)
    return ort.InferenceSession.create(bytes, {
      executionProviders: backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,
      enableMemPattern: false,
    })
  })()
  sessions.set(key, promise)
  promise.catch(() => sessions.delete(key))
  return promise
}

export async function releaseGroup(group: string): Promise<void> {
  const keys = [...sessions.keys()].filter((key) => key.startsWith(`${group}/`))
  for (const key of keys) {
    const promise = sessions.get(key)
    sessions.delete(key)
    try {
      await (await promise)?.release()
    } catch {
      continue
    }
  }
}

export { ort }
