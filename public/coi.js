if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting())
  self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()))
  self.addEventListener('fetch', (e) => {
    const r = e.request
    if (r.cache === 'only-if-cached' && r.mode !== 'same-origin') return
    e.respondWith(
      fetch(r).then((res) => {
        if (res.status === 0) return res
        const h = new Headers(res.headers)
        h.set('Cross-Origin-Embedder-Policy', 'require-corp')
        h.set('Cross-Origin-Opener-Policy', 'same-origin')
        return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h })
      }),
    )
  })
} else if ('serviceWorker' in navigator && !window.crossOriginIsolated) {
  navigator.serviceWorker.register(document.currentScript.src).then((reg) => {
    if (reg.active && !navigator.serviceWorker.controller) window.location.reload()
  })
}
