import { pickBackend } from '../src/engine/backend'
import { readFileSync } from 'node:fs'
import assert from 'node:assert/strict'
import * as ort from 'onnxruntime-web'
import { strainFeatures, predictStrain, flowMetrics, transitionFeatures, selectFlowCandidate, type StrainModel } from '../src/engine/flow'
const root=process.argv[2]??'/mnt/storage/BeatSaberModelTrainer/runs/export/flow-browser'
const model=JSON.parse(readFileSync(`${root}/strain.json`,'utf8')) as StrainModel
const fixture=JSON.parse(readFileSync(`${root}/parity.json`,'utf8'))
let worstFeature=0,worstProbability=0,worstMetric=0
for(const c of fixture.cases) {
  const X=strainFeatures(c.notes,c.spb,c.difficulty,model)
  for(let i=0;i<X.length;i++) for(let j=0;j<model.features.length;j++) {
    const want=c.features[i][j], got=X[i][j]
    assert.equal(Number.isNaN(got),want===null,`${c.name} ${i} ${model.features[j]} missing mismatch: ${got} vs ${want}`)
    if(want!==null) {
      const delta=Math.abs(got-want); worstFeature=Math.max(worstFeature,delta)
      assert.ok(delta<1e-4,`${c.name} ${i} ${model.features[j]}: ${got} vs ${want}`)
    }
  }
  const p=predictStrain(model,X)
  p.forEach((v,i)=>{const e=Math.abs(v-c.probabilities[i]);worstProbability=Math.max(worstProbability,e);assert.ok(e<1e-7,`${c.name} probability ${i}: ${e}`)})
  const m=flowMetrics(c.notes,c.difficulty,model,{...c,energy:Float32Array.from(c.energy)})
  for(const [k,w] of Object.entries(c.metrics)) {const e=Math.abs(m[k as keyof typeof m]-(w as number));worstMetric=Math.max(worstMetric,e);assert.ok(e<1e-7,`${k}: ${e}`)}
}
const t=fixture.transition,X=transitionFeatures(t.swings,t.hand,t.time,t.njs,t.theta)
for(let i=0;i<X.length;i++) assert.ok(Math.abs(X[i]-t.features[Math.floor(i/32)][i%32])<1e-6,`transition ${i}`)
ort.env.wasm.numThreads=1
const sess=await ort.InferenceSession.create(readFileSync(`${root}/critic.onnx`),{executionProviders:['wasm']})
const out=await sess.run({features:new ort.Tensor('float32',X,[96,32])})
const worstCost=Math.max(...Array.from(out.cost.data as Float32Array,(v,i)=>Math.abs(v-t.cost[i])))
assert.ok(worstCost<1e-4,`critic ${worstCost}`)
await sess.release()
const row={notes:300,nps:3,strain_mean:.2,peak_strain:.3,coverage_loss:0,loud_gap_seconds:0}
assert.equal(selectFlowCandidate([row,{...row,nps:.1,strain_mean:0,peak_strain:0},{...row,strain_mean:.15}]),2)
assert.equal(selectFlowCandidate([row,{...row,coverage_loss:5,strain_mean:0,peak_strain:0}]),0)
assert.equal(selectFlowCandidate([row,{...row,loud_gap_seconds:5,strain_mean:0,peak_strain:0}]),0)
console.log(JSON.stringify({cases:fixture.cases.length,worstFeature,worstProbability,worstMetric,worstCost}))

const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator')
try {
  for (const [features, fallback, expected] of [[[], false, 'wasm'], [['shader-f16'], true, 'wasm'], [['shader-f16'], false, 'webgpu']] as const) {
    Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: { requestAdapter: async () => ({ features: new Set(features), isFallbackAdapter: fallback }) } } })
    assert.equal(await pickBackend(), expected)
  }
} finally { if (descriptor) Object.defineProperty(globalThis, 'navigator', descriptor) }
