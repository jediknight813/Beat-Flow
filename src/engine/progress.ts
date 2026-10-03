import type { Progress } from './types'

// Overall pipeline completion, rather than the current stage's percentage.
// This follows runPipeline's order; it is not an elapsed-time estimate.
const stageIndex: Record<Progress['stage'], number> = {
  decode: 0,
  stems: 1,
  beats: 2,
  vocals: 3,
  attacks: 4,
  grid: 5,
  sections: 6,
  tokens: 7,
  candidates: 8,
  walls: 9,
  notes: 10,
  lights: 11,
  style: 12,
  package: 13,
}
const stageCount = Object.keys(stageIndex).length

export function overallGenerationPercent(progress: Progress): number {
  const fraction = Number.isFinite(progress.fraction) ? Math.max(0, Math.min(1, progress.fraction)) : 0
  // Only a successful result means completion; packaging may still be returning.
  return Math.min(99, Math.floor(((stageIndex[progress.stage] + fraction) / stageCount) * 100))
}
