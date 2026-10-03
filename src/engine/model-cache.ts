export type ModelManifest = {
  name: string
  files: string[]
  bytes: number
}

const CACHE = 'beatflow-models-v1'

export async function cachedFetch(url: string, version: string, signal?: AbortSignal): Promise<Response> {
  signal?.throwIfAborted()
  const key = `${url}?v=${version}`
  let cache: Cache | null = null
  try {
    cache = await caches.open(CACHE)
    const hit = await cache.match(key)
    if (hit) { signal?.throwIfAborted(); return hit }
  } catch {
    cache = null
  }
  signal?.throwIfAborted()
  const res = await fetch(url, { signal })
  if (!res.ok) throw new Error(`model chunk missing: ${url}`)
  if (!cache) return res
  // Store, then read back from the cache. Putting a clone while holding the
  // original stalls on large network bodies until the other branch is read.
  try {
    await cache.put(key, res)
    const stored = await cache.match(key)
    if (stored) return stored
  } catch {
    // Quota or storage errors: fall through and fetch uncached.
  }
  signal?.throwIfAborted()
  return fetch(url, { signal })
}

export async function cachedModels(): Promise<string[]> {
  try {
    const cache = await caches.open(CACHE)
    return (await cache.keys()).map((r) => r.url)
  } catch {
    return []
  }
}

export async function clearModelCache(): Promise<void> {
  await caches.delete(CACHE)
}

export async function cacheModelFile(group: string, manifest: ModelManifest, onBytes?: (bytes: number) => void, signal?: AbortSignal): Promise<void> {
  const base = `${import.meta.env.BASE_URL}models/${group}/`
  let done = 0
  for (const file of manifest.files) {
    const res = await cachedFetch(`${base}${file}`, String(manifest.bytes), signal)
    done += (await res.arrayBuffer()).byteLength
    signal?.throwIfAborted()
    onBytes?.(done)
  }
}
