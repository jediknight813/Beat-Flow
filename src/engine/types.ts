export type Difficulty = 'Expert' | 'ExpertPlus'

export type Note = {
  time: number
  x: number
  y: number
  hand: 0 | 1
  direction: number
  angle: number
}

export type Wall = {
  time: number
  duration: number
  x: number
  y: number
  width: number
  height: number
}

export type Arc = {
  time: number
  tailTime: number
  hand: 0 | 1
  x: number
  y: number
  direction: number
  tailX: number
  tailY: number
  tailDirection: number
}

export type Chart = {
  difficulty: Difficulty
  notes: Note[]
  walls: Wall[]
  arcs: Arc[]
}

export type StageName =
  | 'decode'
  | 'stems'
  | 'beats'
  | 'vocals'
  | 'attacks'
  | 'grid'
  | 'sections'
  | 'tokens'
  | 'candidates'
  | 'notes'
  | 'lights'
  | 'style'
  | 'package'

export type Progress = {
  stage: StageName
  fraction: number
  detail?: string
}

export type WallSetting = 'auto' | 'off' | 'light' | 'normal' | 'heavy'
export type LightingSetting = 'auto' | 'calm' | 'normal' | 'intense'
export type Palette = { left: string; right: string; lightA: string; lightB: string }

export type Settings = {
  difficulties: Difficulty[]
  walls: WallSetting
  arcs: boolean
  lighting: LightingSetting
  environment: string
  colors: 'auto' | Palette
  title: string
  artist: string
  cover: 'auto' | Blob
  seed: number
  candidates: number
}

export type CandidateSummary = { seed: number; styleBucket: number; topP: number; notes: number; nps: number; strain_mean: number; peak_strain: number; coverage_loss: number; loud_gap_seconds: number }

export type ChartSummary = {
  difficulty: Difficulty
  notes: number
  arcs: number
  walls: number
  lightEvents: number
  nps: number
  picked: number
  candidates: CandidateSummary[]
}

export type Detected = {
  title: string
  artist: string
  cover: boolean
  bpm: number
  duration: number
  sections: number
  walls: { enabled: boolean; perMinute: number }
  lighting: Exclude<LightingSetting, 'auto'>
  environment: string
  palette: Palette
  seed: number
  charts: ChartSummary[]
}
