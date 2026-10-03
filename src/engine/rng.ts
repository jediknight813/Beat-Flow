export class Rng {
  private state: Uint32Array

  constructor(seed: number) {
    this.state = new Uint32Array(4)
    let s = seed >>> 0
    for (let i = 0; i < 4; i++) {
      s = (s + 0x9e3779b9) >>> 0
      let z = s
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0
      this.state[i] = (z ^ (z >>> 16)) >>> 0
    }
  }

  next(): number {
    const s = this.state
    const result = (Math.imul(s[1], 5) << 7) | (Math.imul(s[1], 5) >>> 25)
    const t = s[1] << 9
    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]
    s[2] ^= t
    s[3] = (s[3] << 11) | (s[3] >>> 21)
    return (Math.imul(result, 9) >>> 0) / 4294967296
  }

  random(): number {
    return this.next()
  }

  uniform(a: number, b: number): number {
    return a + (b - a) * this.next()
  }

  integer(n: number): number {
    return Math.min(n - 1, Math.floor(this.next() * n))
  }

  choice(prob: number[]): number {
    let u = this.next()
    for (let i = 0; i < prob.length; i++) {
      u -= prob[i]
      if (u < 0) return i
    }
    return prob.length - 1
  }

  gumbel(): number {
    return -Math.log(-Math.log(Math.max(this.next(), 1e-12)))
  }
}
