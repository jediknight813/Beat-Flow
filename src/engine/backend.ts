export type Backend = 'webgpu' | 'wasm'

export async function pickBackend(): Promise<Backend> {
  const gpu = (navigator as Navigator & { gpu?: unknown }).gpu
  if (!gpu) return 'wasm'
  try {
    const adapter = await (gpu as { requestAdapter: () => Promise<unknown> }).requestAdapter()
    return adapter ? 'webgpu' : 'wasm'
  } catch {
    return 'wasm'
  }
}

