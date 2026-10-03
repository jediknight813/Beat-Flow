export class MT19937 {
  private mt = new Uint32Array(624)
  private index = 624

  constructor(seed: number) {
    this.mt[0] = seed >>> 0
    for (let i = 1; i < 624; i++) {
      const prev = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30)
      this.mt[i] = (Math.imul(1812433253, prev) + i) >>> 0
    }
  }

  private twist(): void {
    const mt = this.mt
    for (let i = 0; i < 624; i++) {
      const y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff)
      mt[i] = mt[(i + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0)
    }
    this.index = 0
  }

  uint32(): number {
    if (this.index >= 624) this.twist()
    let y = this.mt[this.index++]
    y ^= y >>> 11
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= y >>> 18
    return y >>> 0
  }

  random(): number {
    const a = this.uint32() >>> 5
    const b = this.uint32() >>> 6
    return (a * 67108864 + b) / 9007199254740992
  }
}
