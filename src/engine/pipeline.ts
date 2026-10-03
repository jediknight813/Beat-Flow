import { getBackendPreference, type BackendPreference } from './backend'
import type { Progress, Settings } from './types'
import type { CoreResult } from './core'
import type { WorkerRequest, WorkerResponse } from './worker'
import { decodeAudio } from './audio'

export type Result = CoreResult

let worker: Worker | null = null
let nextId = 0
let workerPreference: BackendPreference | null = null

export function resetGenerationWorker(): void {
  worker?.terminate()
  worker = null
  workerPreference = null
}

function pipelineWorker(preference: BackendPreference): Worker {
  if (workerPreference !== preference) resetGenerationWorker()
  workerPreference = preference
  worker ??= new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
  return worker
}

function cancelled(): DOMException {
  return new DOMException('cancelled', 'AbortError')
}

export async function generate(file: File, settings: Settings, onProgress: (p: Progress) => void, signal?: AbortSignal): Promise<Result> {
  onProgress({ stage: 'decode', fraction: 0 })
  const audio = await decodeAudio(file)
  if (signal?.aborted) throw cancelled()
  onProgress({ stage: 'decode', fraction: 1, detail: `${audio.duration.toFixed(1)} s at ${audio.sampleRate} Hz` })
  const bytes = await file.arrayBuffer()
  const [left, right] = audio.stereo44k
  const id = ++nextId
  const backendPreference = getBackendPreference()
  const w = pipelineWorker(backendPreference)
  return new Promise<Result>((resolve, reject) => {
    const finish = () => {
      w.removeEventListener('message', onMessage)
      w.removeEventListener('error', onError)
      signal?.removeEventListener('abort', onAbort)
    }
    const onMessage = (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data
      if (msg.id !== id) return
      if (msg.type === 'progress') onProgress(msg.progress)
      else {
        finish()
        if (msg.type === 'done') resolve(msg.result)
        else reject(new Error(msg.message))
      }
    }
    const onError = (event: ErrorEvent) => {
      finish()
      w.terminate()
      if (worker === w) worker = null
      reject(new Error(event.message || 'The generator stopped unexpectedly'))
    }
    const onAbort = () => {
      finish()
      w.terminate()
      if (worker === w) worker = null
      reject(cancelled())
    }
    w.addEventListener('message', onMessage)
    w.addEventListener('error', onError)
    signal?.addEventListener('abort', onAbort, { once: true })
    const request: WorkerRequest = {
      type: 'generate',
      backendPreference,
      id,
      audio: { left, right, sampleRate: audio.sampleRate },
      file: { name: file.name, bytes },
      settings,
    }
    w.postMessage(request, [left.buffer, right.buffer, bytes])
  })
}
