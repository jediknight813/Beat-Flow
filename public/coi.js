// GitHub Pages can't send COOP/COEP headers, so this service worker adds them to
// make the page cross-origin isolated (SharedArrayBuffer for threaded inference).
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
      // A failed or aborted request (offline, page reload, worker update) is a
      // normal network error, not an uncaught exception in the worker.
      }).catch(() => Response.error()),
    )
  })
} else if ('serviceWorker' in navigator && !window.crossOriginIsolated) {
  // Isolation is decided when the document loads, so reload once the worker
  // controls the page. The session flag stops a loop if isolation never takes.
  const reload = () => {
    try {
      if (sessionStorage.getItem('coi-reloaded')) return
      sessionStorage.setItem('coi-reloaded', '1')
    } catch {
      return
    }
    window.location.reload()
  }
  navigator.serviceWorker.addEventListener('controllerchange', reload)
  navigator.serviceWorker.register(document.currentScript.src).then((reg) => {
    if (reg.active && !navigator.serviceWorker.controller) reload()
  })
} else if (window.crossOriginIsolated) {
  try { sessionStorage.removeItem('coi-reloaded') } catch { /* storage unavailable */ }
}
