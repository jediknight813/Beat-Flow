export type Audio = {
  stereo44k: [Float32Array, Float32Array]
  sampleRate: 44100
  duration: number
}

async function decode(file: File, sampleRate: number): Promise<AudioBuffer> {
  const ctx = new OfflineAudioContext(2, 1, sampleRate)
  return ctx.decodeAudioData(await file.arrayBuffer())
}

export async function decodeAudio(file: File): Promise<Audio> {
  const decoded = await decode(file, 44100)
  const left = decoded.getChannelData(0).slice()
  const right = decoded.numberOfChannels > 1 ? decoded.getChannelData(1).slice() : left.slice()
  return { stereo44k: [left, right], sampleRate: 44100, duration: decoded.duration }
}
