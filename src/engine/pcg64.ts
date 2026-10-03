const MASK32 = 0xffffffffn
const MASK64 = 0xffffffffffffffffn
const MASK128 = (1n << 128n) - 1n
const MULT = (2549297995355413924n << 64n) | 4865540595714422341n

function seedWords(seed: number): number[] {
  const words: number[] = []
  let s = BigInt(Math.max(0, Math.floor(seed)))
  if (s === 0n) return [0]
  while (s > 0n) {
    words.push(Number(s & MASK32))
    s >>= 32n
  }
  return words
}

function seedState(seed: number): bigint[] {
  const entropy = seedWords(seed)
  const pool = new Uint32Array(4)
  let hashConst = 0x43b0d7e5
  const hashmix = (value: number) => {
    value = (value ^ hashConst) >>> 0
    hashConst = Math.imul(hashConst, 0x931e8875) >>> 0
    value = Math.imul(value, hashConst) >>> 0
    return (value ^ (value >>> 16)) >>> 0
  }
  const mix = (x: number, y: number) => {
    const r = (Math.imul(0xca01f9dd, x) - Math.imul(0x4973f715, y)) >>> 0
    return (r ^ (r >>> 16)) >>> 0
  }
  for (let i = 0; i < 4; i++) pool[i] = hashmix(i < entropy.length ? entropy[i] : 0)
  for (let src = 0; src < 4; src++) for (let dst = 0; dst < 4; dst++) if (src !== dst) pool[dst] = mix(pool[dst], hashmix(pool[src]))
  for (let src = 4; src < entropy.length; src++) for (let dst = 0; dst < 4; dst++) pool[dst] = mix(pool[dst], hashmix(entropy[src]))
  let hc = 0x8b51f9dd
  const words: number[] = []
  for (let i = 0; i < 8; i++) {
    let v = (pool[i % 4] ^ hc) >>> 0
    hc = Math.imul(hc, 0x58f38ded) >>> 0
    v = Math.imul(v, hc) >>> 0
    words.push((v ^ (v >>> 16)) >>> 0)
  }
  const out: bigint[] = []
  for (let i = 0; i < 4; i++) out.push((BigInt(words[2 * i + 1]) << 32n) | BigInt(words[2 * i]))
  return out
}

export class Pcg64 {
  private state: bigint
  private inc: bigint
  private spare = -1

  constructor(seed: number) {
    const [s0, s1, i0, i1] = seedState(seed)
    const initState = (s0 << 64n) | s1
    const initInc = (i0 << 64n) | i1
    this.state = 0n
    this.inc = ((initInc << 1n) | 1n) & MASK128
    this.step()
    this.state = (this.state + initState) & MASK128
    this.step()
  }

  private step() {
    this.state = (this.state * MULT + this.inc) & MASK128
  }

  next64(): bigint {
    this.step()
    const s = this.state
    const rot = Number(s >> 122n)
    const x = ((s >> 64n) ^ s) & MASK64
    return rot ? ((x >> BigInt(rot)) | (x << BigInt(64 - rot))) & MASK64 : x
  }

  next32(): number {
    if (this.spare >= 0) {
      const v = this.spare
      this.spare = -1
      return v
    }
    const v = this.next64()
    this.spare = Number(v >> 32n)
    return Number(v & MASK32)
  }

  random(): number {
    return Number(this.next64() >> 11n) / 9007199254740992
  }

  bounded(max: number): number {
    if (max === 0) return 0
    const range = max + 1
    let m = BigInt(this.next32()) * BigInt(range)
    let leftover = Number(m & MASK32)
    if (leftover < range) {
      const threshold = (0x100000000 - range) % range
      while (leftover < threshold) {
        m = BigInt(this.next32()) * BigInt(range)
        leftover = Number(m & MASK32)
      }
    }
    return Number(m >> 32n)
  }

  choice(population: number, count: number): number[] {
    const idx = new Array<number>(count)
    const size = 1 << Math.ceil(Math.log2(Math.max(2, 1.2 * count)))
    const mask = size - 1
    const set = new Array<number>(size).fill(-1)
    for (let j = population - count; j < population; j++) {
      const val = this.bounded(j)
      let loc = val & mask
      while (set[loc] !== -1 && set[loc] !== val) loc = (loc + 1) & mask
      if (set[loc] === -1) {
        set[loc] = val
        idx[j - population + count] = val
      } else {
        loc = j & mask
        while (set[loc] !== -1) loc = (loc + 1) & mask
        set[loc] = j
        idx[j - population + count] = j
      }
    }
    for (let i = count - 1; i >= 1; i--) {
      const j = this.bounded(i)
      const t = idx[i]
      idx[i] = idx[j]
      idx[j] = t
    }
    return idx
  }
}
