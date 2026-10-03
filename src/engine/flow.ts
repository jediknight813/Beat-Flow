import * as ort from 'onnxruntime-web'
import type { Note, Difficulty } from './types'
import type { Swing } from './notes'
import { analyzeSwings } from './judge'
import { roundDecimals } from './dsp'

export type StrainModel = {
  schema: number; features: string[]; baseline: number; trees: number[][][]
  theta: number; referenceSkillPercentile: number; sourceSha256: string
}
export type FlowMetrics = { strain_mean: number; peak_strain: number; coverage_loss: number; loud_gap_seconds: number }
const vectors = [[0, 1], [0, -1], [-1, 0], [1, 0], [-1, 1], [1, 1], [-1, -1], [1, -1]]
const unit = (v: number[]) => { const m = Math.hypot(...v); return v.map(x => x / m) }
const median = (x: number[]) => { const a = [...x].sort((a,b)=>a-b), k = a.length >> 1; return a.length ? a.length % 2 ? a[k] : (a[k-1]+a[k])/2 : 0 }
const mean = (x: number[]) => x.reduce((a,b)=>a+b,0)/x.length
function bisect(a: number[], x: number, right = false): number {
  let lo=0, hi=a.length
  while(lo<hi) { const m=(lo+hi)>>1; if(a[m]<x || (right && a[m]===x)) lo=m+1; else hi=m }
  return lo
}

// Exact feature order used by bsmapper.v8.critic.features_one, then player skill.
export function transitionFeatures(swings: Swing[], hand: number, time: number, njs: number, theta: number): Float32Array {
  const mine=swings.filter(s=>s.hand===hand), others=swings.filter(s=>s.hand!==hand)
  const prev=mine.at(-1), pp=mine.at(-2), other=others.at(-1)
  const pc=prev ? unit(vectors[prev.cut]) : [0,0], ppc=pp ? unit(vectors[pp.cut]) : [0,0], oc=other ? unit(vectors[other.cut]) : [0,0]
  const out=new Float32Array(96*32)
  for(let cut=0;cut<8;cut++) for(let cell=0;cell<12;cell++) {
    const [cx,cy]=unit(vectors[cut]), x=Math.floor(cell/3), y=cell%3
    const dx=prev ? x-cx-(Math.floor(prev.cell/3)+pc[0]) : 0, dy=prev ? y-cy-(prev.cell%3+pc[1]) : 0
    const dt=other ? time-other.time : 4
    const row=[hand,Math.log(prev ? Math.max(time-prev.time,0.001):4), pp&&prev ? Math.log(Math.max(prev.time-pp.time,0.001)):0,
      prev?Math.floor(prev.cell/3):0,prev?prev.cell%3:0,...pc,prev?Number(prev.notes[0].direction===8):0,prev?Math.min(3,prev.notes.length-1):0,
      x,y,cx,cy,0,0,dx,dy,Math.hypot(dx,dy),pc[0]*cx+pc[1]*cy,pc[0]*cy-pc[1]*cx,...ppc,Number(!!prev),
      Math.log(Math.max(dt,0.001)),other?Math.floor(other.cell/3):0,other?other.cell%3:0,...oc,Number(!!other&&Math.abs(dt)<=0.03),
      Number(!!other&&(hand===0 ? x>Math.floor(other.cell/3):x<Math.floor(other.cell/3))),njs/20,theta]
    out.set(row,(cut*12+cell)*32)
  }
  return out
}

export async function flowCosts(session: ort.InferenceSession, swings: Swing[], hand: number, time: number, difficulty: Difficulty): Promise<Float32Array> {
  const X=transitionFeatures(swings,hand,time,difficulty==='Expert'?16:18,difficulty==='Expert'?-1.04:-0.78)
  const out=await session.run({features:new ort.Tensor('float32',X,[96,32])})
  const tensor=out.cost
  const costs=Float32Array.from(tensor.data as Float32Array)
  tensor.dispose()
  return costs
}

export function strainFeatures(notes: Note[], spb: number, difficulty: Difficulty, model: Pick<StrainModel,'features'|'theta'>): Float32Array[] {
  if(!notes.length) return []
  const rows: Record<string,number>[]=notes.map(()=>({})), idx=new Map(notes.map((n,i)=>[n,i]))
  const times=notes.map(n=>n.time), st=[...times].sort((a,b)=>a-b), end=Math.max(st.at(-1)!,1), span=Math.max(st.at(-1)!-st[0],1)
  const ut=[...new Set(times.map(t=>roundDecimals(t,3)))].sort((a,b)=>a-b)
  const anyGap=new Map(ut.map((t,i)=>[t,i?t-ut[i-1]:NaN]))
  notes.forEach((n,i)=>{
    const r=rows[i], gap=anyGap.get(roundDecimals(n.time,3))!
    Object.assign(r,{diff:difficulty==='Expert'?3:4,njs:difficulty==='Expert'?16:18,rel_pos:n.time/end,chart_nps:notes.length/span,
      beats_per_sec:1/spb,x:n.x,y:n.y,dot:Number(n.direction===8),dir_class:[0,1,2,2,3,3,4,4,5][n.direction],any_gap:gap,after_rest:Number(gap>=1),theta:model.theta})
    for(const w of [2,10,30]) r[`nps${w}`]=(bisect(st,n.time,true)-bisect(st,n.time-w))/w
  })
  const hands=analyzeSwings(notes)
  for(const h of [0,1]) {
    const sw=hands[h], ost=hands[1-h].map(s=>s.start); let run=0
    sw.forEach((s,k)=>{
      const g=s.gap??NaN; run=Number.isFinite(g)&&g<=0.3 ? run+1:0
      const travel=k ? Math.hypot(s.entry[0]-sw[k-1].exit[0],s.entry[1]-sw[k-1].exit[1]):NaN
      s.notes.forEach((n,rank)=>{
        const r=rows[idx.get(n)!]
        Object.assign(r,{hand_gap:rank?0:g,beat_gap:rank?0:g/spb,travel,speed:s.speed??NaN,redirect:s.redirect??NaN,
          parity_break:Number(s.reset==='break'),rest_reset:Number(s.reset==='rest'),roll_abs:Math.abs(s.roll),crossover:Number(!!s.crossover),paths_cross:Number(!!s.paths),
          in_swing:rank,swing_len:s.notes.length,swing_rank:k,stream_pos:run})
        if(ost.length) { const j=bisect(ost,s.start); r.double=Number(Math.min(j?Math.abs(ost[j-1]-s.start):9,j<ost.length?Math.abs(ost[j]-s.start):9)<=0.05); r.other_hand_gap=j?s.start-ost[j-1]:NaN }
        if(k) { const p=rows[idx.get(sw[k-1].notes[0])!]; for(const f of ['hand_gap','travel','redirect','speed','roll_abs']) r[`prev_${f}`]=p[f] }
      })
    })
  }
  let count=0, last: number|null=null
  for(const i of times.map((_,i)=>i).sort((a,b)=>times[a]-times[b]||a-b)) {
    const r=rows[i]; if(r.any_gap>=1||Number.isNaN(r.any_gap)) count=0; else if(last!==null&&times[i]-times[last]>0.01) count++
    r.first_after_rest_n=count; last=i
  }
  return rows.map(r=>Float32Array.from(model.features,f=>r[f]??NaN))
}

export function predictStrain(model: StrainModel, rows: Float32Array[]): number[] {
  if(model.schema!==1) throw new Error('Unsupported strain model schema')
  return rows.map(row=>{
    let sum=model.baseline
    for(const tree of model.trees) {
      let i=0
      while(tree[i][0]>=0) { const n=tree[i], v=row[n[0]]; i=Number.isNaN(v)?(n[4]?n[2]:n[3]):v<=n[1]?n[2]:n[3] }
      sum+=tree[i][5]
    }
    return 1/(1+Math.exp(-sum))
  })
}

export function flowMetrics(notes: Note[], difficulty: Difficulty, model: StrainModel, facts: {spb:number; musicStart:number; musicEnd:number; energy:Float32Array}): FlowMetrics {
  const ns=[...notes].sort((a,b)=>a.time-b.time||a.hand-b.hand), times=ns.map(n=>n.time)
  if(ns.length<30) return {strain_mean:NaN,peak_strain:NaN,coverage_loss:Infinity,loud_gap_seconds:Infinity}
  const p=predictStrain(model,strainFeatures(ns,facts.spb,difficulty,model)), cs=[0]
  for(const x of p) cs.push(cs.at(-1)!+x)
  let peak=-Infinity, peakMean=0
  times.forEach((t,i)=>{ const hi=bisect(times,t+10,true), sum=cs[hi]-cs[i]; if(sum>peak) {peak=sum;peakMean=sum/(hi-i)} })
  const threshold=0.5*median(Array.from(facts.energy).filter(x=>x>0)); let loud=0
  for(let i=1;i<times.length;i++) if(times[i]-times[i-1]>2) {
    const lo=Math.max(0,Math.floor((times[i-1]+0.5)*50)), hi=Math.min(facts.energy.length,Math.floor((times[i]-0.5)*50))
    for(let f=lo;f<hi;f++) if(facts.energy[f]>threshold) loud+=1/50
  }
  return {strain_mean:mean(p),peak_strain:peakMean,coverage_loss:Math.max(0,times[0]-facts.musicStart-0.75)+Math.max(0,facts.musicEnd-times.at(-1)!-0.75),loud_gap_seconds:loud}
}

export function selectFlowCandidate(rows: ({notes:number;nps:number}&FlowMetrics)[]): number {
  const good=rows.map((r,i)=>({r,i})).filter(({r})=>r.notes>=30&&[r.nps,r.strain_mean,r.peak_strain,r.coverage_loss,r.loud_gap_seconds].every(Number.isFinite))
  if(!good.length) throw new Error('No complete candidates with finite flow scores')
  const mid=median(good.map(({r})=>r.nps))
  let eligible=good.filter(({r})=>r.nps>=0.85*mid&&r.nps<=1.15*mid)
  if(!eligible.length) {const delta=Math.min(...good.map(({r})=>Math.abs(r.nps-mid)));eligible=good.filter(({r})=>Math.abs(r.nps-mid)<=delta+1e-9)}
  const coverage=Math.min(...eligible.map(({r})=>r.coverage_loss)); eligible=eligible.filter(({r})=>r.coverage_loss<=coverage+0.5)
  const gaps=Math.min(...eligible.map(({r})=>r.loud_gap_seconds)); eligible=eligible.filter(({r})=>r.loud_gap_seconds<=gaps+1)
  return eligible.reduce((a,b)=>b.r.strain_mean+b.r.peak_strain<a.r.strain_mean+a.r.peak_strain?b:a).i
}
