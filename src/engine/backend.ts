export type Backend = 'webgpu' | 'wasm'

type Adapter = { features: { has(name: string): boolean }; isFallbackAdapter?: boolean }
type GPU = { requestAdapter(): Promise<Adapter | null> }

export async function pickBackend(): Promise<Backend> {
  const gpu = (navigator as unknown as { gpu?: GPU }).gpu
  if (!gpu) return 'wasm'
  try {
    const adapter = await gpu.requestAdapter()
    // The compact decoder uses FP16. Without native support, the WebGPU route
    // copies between providers and was much slower than the INT8 WASM route.
    return adapter && !adapter.isFallbackAdapter && adapter.features.has('shader-f16') ? 'webgpu' : 'wasm'
  } catch {
    return 'wasm'
  }
}
