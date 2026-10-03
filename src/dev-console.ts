// React's optional extension advertisement is informational, not a diagnostic.
// Hide that one startup message; preserve every warning, error, and other log.
if (import.meta.env.DEV) {
  const info = console.info
  console.info = (...args: unknown[]) => {
    if (typeof args[0] === 'string' && args[0].startsWith('%cDownload the React DevTools for a better development experience:')) {
      console.info = info
      return
    }
    info.apply(console, args)
  }
}
