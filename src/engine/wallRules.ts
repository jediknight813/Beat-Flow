import type { Note, Wall } from './types'

export const CROUCH_CLEAR = 0.35
export const CROUCH_STAND = 0.9
export const DODGE_BEFORE = 0.4
export const DODGE_AFTER = 0.6

export type WallKind = 'crouch' | 'dodge' | 'side' | 'other'

export function wallKind(w: Wall): WallKind {
  if (w.y >= 2 && w.height <= 3) return 'crouch'
  if (w.y === 0 && w.height >= 3) {
    const lanes = laneSet(w)
    return lanes.has(1) || lanes.has(2) ? 'dodge' : 'side'
  }
  return 'other'
}

function laneSet(w: Wall): Set<number> {
  const s = new Set<number>()
  for (let x = w.x; x < w.x + Math.max(1, w.width); x++) s.add(x)
  return s
}

function rowSet(w: Wall): Set<number> {
  const s = new Set<number>()
  for (let y = w.y; y < Math.min(3, w.y + w.height); y++) s.add(y)
  return s
}

function farSide(lanes: Set<number>): Set<number> {
  const min = Math.min(...lanes)
  const max = Math.max(...lanes)
  const open = [0, 1, 2, 3].filter((x) => !lanes.has(x))
  const leftOpen = open.filter((x) => x < min).length
  const rightOpen = open.filter((x) => x > max).length
  const far = new Set<number>()
  if (rightOpen >= leftOpen) for (let x = 0; x <= max; x++) far.add(x)
  else for (let x = min; x < 4; x++) far.add(x)
  return far
}

export function fairWall(w: Wall, notes: Note[], placed: Wall[]): boolean {
  const t0 = w.time
  const t1 = w.time + w.duration
  const lanes = laneSet(w)
  const rows = rowSet(w)
  const k = wallKind(w)
  const blocked = k === 'dodge' ? farSide(lanes) : new Set<number>()
  for (const n of notes) {
    if (n.time < t0 - Math.max(0.25, DODGE_BEFORE)) continue
    if (n.time > t1 + Math.max(CROUCH_STAND, DODGE_AFTER)) break
    const during = t0 - 0.25 <= n.time && n.time <= t1 + 0.15
    if (during && lanes.has(n.x) && rows.has(n.y)) return false
    if (k === 'dodge' && t0 - DODGE_BEFORE <= n.time && n.time <= t1 + DODGE_AFTER && blocked.has(n.x)) return false
    if (k === 'crouch') {
      if (during && n.y === 2) return false
      if (t1 < n.time && n.time <= t1 + CROUCH_CLEAR) return false
      if (t1 < n.time && n.time <= t1 + CROUCH_STAND && n.y >= 1) return false
    }
  }
  for (const p of placed) {
    const pk = wallKind(p)
    const gap = t0 - (p.time + p.duration)
    if ((k === 'dodge' || k === 'crouch') && (pk === 'dodge' || pk === 'crouch') && -p.duration < gap && gap < 0.6) return false
    if (t0 < p.time + p.duration && p.time < t1) {
      const pl = laneSet(p)
      for (const x of lanes) if (pl.has(x)) return false
      const union = new Set([...lanes, ...pl].filter((x) => x >= 0 && x <= 3))
      if (union.size >= 3) return false
    }
  }
  if (k === 'crouch' && w.duration > 2) return false
  return true
}

export function wallNoteMask(walls: Wall[], time: number): { blocked: boolean[]; all: boolean } {
  const blocked = new Array<boolean>(12).fill(false)
  for (const w of walls) {
    const t0 = w.time
    const t1 = w.time + w.duration
    if (time < t0 - Math.max(0.25, DODGE_BEFORE) || time > t1 + Math.max(CROUCH_STAND, DODGE_AFTER)) continue
    const k = wallKind(w)
    const lanes = laneSet(w)
    const rows = rowSet(w)
    const during = t0 - 0.25 <= time && time <= t1 + 0.15
    for (let c = 0; c < 12; c++) {
      const x = Math.floor(c / 3)
      const y = c % 3
      if (during && lanes.has(x) && rows.has(y)) blocked[c] = true
      if (k === 'crouch' && ((during && y === 2) || (t1 < time && time <= t1 + CROUCH_CLEAR) || (t1 < time && time <= t1 + CROUCH_STAND && y >= 1))) blocked[c] = true
    }
    if (k === 'dodge' && t0 - DODGE_BEFORE <= time && time <= t1 + DODGE_AFTER) {
      const far = farSide(lanes)
      for (let c = 0; c < 12; c++) if (far.has(Math.floor(c / 3))) blocked[c] = true
    }
  }
  return { blocked, all: blocked.every(Boolean) }
}
