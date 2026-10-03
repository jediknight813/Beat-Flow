import { useEffect, useRef } from 'react'

export default function Backdrop() {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    let cancelled = false
    let dispose: (() => void) | undefined
    // Let the controls paint before loading and constructing the 3D scene.
    void import('./backdrop').then(({ createBackdrop }) => {
      if (!cancelled && ref.current) dispose = createBackdrop(ref.current)
    }).catch((error: unknown) => {
      console.error('Unable to load the background scene:', error)
    })
    return () => { cancelled = true; dispose?.() }
  }, [])
  return <canvas ref={ref} aria-hidden className="fixed inset-0 z-0 h-full w-full" style={{ filter: 'blur(3.6px)' }} />
}
