import type { Progress } from './types'

// Keep labels and overall progress in pipeline order. This measures completion,
// not elapsed time; note generation includes strain checks and selection.
export const generationStages: Record<Progress['stage'], { label: string; detail: string; phase: number }> = {
  decode: { phase: 0, label: 'Reading your song', detail: 'Preparing the audio.' },
  stems: { phase: 0, label: 'Separating instruments and vocals', detail: 'Listening to the drums, bass, vocals, and other instruments.' },
  beats: { phase: 0, label: 'Finding the beat', detail: 'Following the tempo and rhythm.' },
  vocals: { phase: 0, label: 'Following the vocals', detail: 'Finding vocal timing and pitch.' },
  attacks: { phase: 0, label: 'Finding musical accents', detail: 'Picking out sharp hits and changes in the music.' },
  grid: { phase: 0, label: 'Aligning note timing', detail: 'Lining up the rhythm and musical accents.' },
  sections: { phase: 0, label: 'Finding song sections', detail: 'Following changes in the song’s energy.' },
  tokens: { phase: 0, label: 'Preparing the song for mapping', detail: 'Bringing the rhythm, vocals, and energy together.' },
  candidates: { phase: 0, label: 'Finding moments for notes', detail: 'Choosing possible note timings from the music.' },
  notes: { phase: 1, label: 'Creating flowing notes', detail: 'Writing variations, then checking strain and pacing.' },
  lights: { phase: 2, label: 'Creating the light show', detail: 'Matching the lighting to the music and notes.' },
  style: { phase: 3, label: 'Choosing colors and environment', detail: 'Applying your choices and matching automatic settings to the song.' },
  package: { phase: 3, label: 'Preparing your map', detail: 'Saving the notes, lighting, and audio together.' },
}
export const generationPhases = ['Analyze', 'Map', 'Light', 'Finish']

const stageOrder = Object.keys(generationStages) as Progress['stage'][]

export function overallGenerationPercent(progress: Progress): number {
  const fraction = Number.isFinite(progress.fraction) ? Math.max(0, Math.min(1, progress.fraction)) : 0
  // Only a successful result means completion; packaging may still be returning.
  return Math.min(99, Math.floor(((stageOrder.indexOf(progress.stage) + fraction) / stageOrder.length) * 100))
}
