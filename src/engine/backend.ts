export type Backend = 'webgpu' | 'wasm'
export type BackendPreference = 'auto' | 'wasm' | 'gpu-high-performance' | 'gpu-low-power' | 'gpu-default'
type PowerPreference = 'high-performance' | 'low-power'
type Adapter = {
  features: { has(name: string): boolean }
  info?: { vendor?: string; architecture?: string; device?: string; description?: string; isFallbackAdapter?: boolean }
  isFallbackAdapter?: boolean
}
type GPU = { requestAdapter(options?: { powerPreference?: PowerPreference }): Promise<Adapter | null> }
export type BackendChoice = { id: Exclude<BackendPreference, 'auto'>; backend: Backend; label: string; powerPreference?: PowerPreference }
export type BackendSelection = BackendChoice & { adapter?: Adapter }
const CPU: BackendChoice = { id: 'wasm', backend: 'wasm', label: 'CPU' }
const STORAGE_KEY = 'beatflow-compute-preference'
let preference: BackendPreference | undefined
let available: Promise<BackendSelection[]> | null = null

export function getBackendPreference(): BackendPreference {
  if (preference) return preference
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'wasm' || saved === 'gpu-high-performance' || saved === 'gpu-low-power' || saved === 'gpu-default') return preference = saved
  } catch { /* The worker and private browsing may not have localStorage. */ }
  return preference = 'auto'
}

export function setBackendPreference(next: BackendPreference): void {
  preference = next
  try { localStorage.setItem(STORAGE_KEY, next) } catch { /* Keep the session choice. */ }
}

async function discover(): Promise<BackendSelection[]> {
  const gpu = (navigator as unknown as { gpu?: GPU }).gpu
  if (!gpu) return [CPU]
  const choices: BackendSelection[] = []
  const seen = new Set<string>()
  // WebGPU exposes power preferences, not a list of every physical GPU. Ask for
  // the high-performance adapter first, then offer a distinct low-power adapter.
  for (const powerPreference of ['high-performance', 'low-power', undefined] as const) {
    if (!powerPreference && choices.length) break
    try {
      const adapter = await gpu.requestAdapter(powerPreference ? { powerPreference } : undefined)
      // Our FP16 decoder needs native support; mixed-provider fallback is slower
      // than the CPU INT8 model. Software adapters are not a separate GPU choice.
      if (!adapter || adapter.isFallbackAdapter || adapter.info?.isFallbackAdapter || !adapter.features.has('shader-f16')) continue
      const info = adapter.info
      const identity = [info?.vendor, info?.architecture, info?.device, info?.description].filter(Boolean).join('|') || 'undisclosed'
      if (seen.has(identity)) continue
      seen.add(identity)
      const name = info?.description?.trim() || info?.vendor?.trim()
      choices.push({ id: powerPreference ? `gpu-${powerPreference}` : 'gpu-default', backend: 'webgpu', label: name ? `GPU · ${name}` : 'GPU', powerPreference, adapter })
    } catch { /* Try the next adapter preference, then fall back to CPU. */ }
  }
  return [...choices, CPU]
}

function selections(): Promise<BackendSelection[]> { return available ??= discover() }

export async function backendChoices(): Promise<BackendChoice[]> {
  return (await selections()).map(({ adapter: _adapter, ...choice }) => choice)
}

export async function selectBackend(requested = getBackendPreference()): Promise<BackendSelection> {
  const choices = await selections()
  if (requested !== 'auto') return choices.find((choice) => choice.id === requested) ?? CPU
  return choices[0] // high-performance GPU first; CPU when none is compatible.
}

export async function pickBackend(): Promise<Backend> {
  return (await selectBackend()).backend
}
