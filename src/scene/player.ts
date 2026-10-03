import {
  AdditiveBlending, Color, CylinderGeometry, Group, Mesh, MeshBasicMaterial,
  MeshStandardMaterial, Quaternion, Vector3,
  type BufferGeometry, type Material,
} from 'three'
import { createHeadset } from './headset'
import { HIT_Z, NOTE_SPEED, PLAYER_Z } from './layout'

export type ReplayTarget = { position: Vector3; direction: Vector3 }

export function createPlayer() {
  const group = new Group()
  const geometries = new Set<BufferGeometry>()
  const materials = new Set<Material>()
  const addMesh = (parent: Group, geometry: BufferGeometry, material: Material, x = 0, y = 0, z = 0) => {
    geometries.add(geometry)
    materials.add(material)
    const mesh = new Mesh(geometry, material)
    mesh.position.set(x, y, z)
    parent.add(mesh)
    return mesh
  }
  const rubber = new MeshStandardMaterial({ color: 0x11131e, roughness: 0.88 })
  const metal = new MeshStandardMaterial({ color: 0x8996ae, roughness: 0.25, metalness: 0.8 })
  const headsetModel = createHeadset()
  const headset = headsetModel.group
  headset.position.set(0, 1.85, PLAYER_Z + 0.35)
  group.add(headset)

  const sabers = [0xff163e, 0x16baff].map((color, index) => {
    const saber = new Group()
    const glow = new MeshBasicMaterial({
      color: new Color(color).multiplyScalar(3.5), toneMapped: false,
      transparent: true, opacity: 0.65, blending: AdditiveBlending, depthWrite: false,
    })
    const core = new MeshBasicMaterial({ color: new Color(3.5, 3.5, 3.5), toneMapped: false })
    addMesh(saber, new CylinderGeometry(0.065, 0.075, 0.34, 12), rubber)
    addMesh(saber, new CylinderGeometry(0.08, 0.08, 0.065, 12), metal, 0, 0.17)
    addMesh(saber, new CylinderGeometry(0.075, 0.075, 0.055, 12), metal, 0, -0.17)
    addMesh(saber, new CylinderGeometry(0.069, 0.069, 0.035, 12), glow, 0, 0.1)
    addMesh(saber, new CylinderGeometry(0.027, 0.047, 2, 12), glow, 0, 1.2)
    addMesh(saber, new CylinderGeometry(0.012, 0.02, 2.01, 10), core, 0, 1.2)
    saber.position.set(index === 0 ? -0.7 : 0.7, 0.55, PLAYER_Z)
    group.add(saber)
    return saber
  })

  const up = new Vector3(0, 1, 0)
  const aim = new Vector3()
  const swingAim = new Vector3()
  const hand = new Vector3()
  const swingHand = new Vector3()
  const smoothAims = [new Vector3(), new Vector3()]
  const rotation = new Quaternion()
  const followThrough: ({ target: ReplayTarget; age: number } | null)[] = [null, null]
  const ease = (value: number) => {
    const t = Math.max(0, Math.min(1, value))
    return t * t * (3 - 2 * t)
  }
  const pose = (target: ReplayTarget, side: number, offset: number) => {
    swingAim.copy(target.position).addScaledVector(target.direction, offset)
    swingAim.z = HIT_Z
    swingHand.set(side * 0.7 + (target.position.x - side * 0.7) * 0.3,
      0.55 + (target.position.y - 1) * 0.35 + offset * target.direction.y * 0.08,
      PLAYER_Z - Math.sin(Math.abs(offset)) * 0.12)
  }
  const update = (time: number, dt: number, targets: (ReplayTarget | null)[]) => {
    headset.position.y = 1.85 + Math.sin(time * 1.7) * 0.025
    headset.rotation.set(-0.1 + Math.sin(time * 1.2) * 0.025, Math.sin(time * 0.8) * 0.08, Math.sin(time) * 0.025)
    sabers.forEach((saber, index) => {
      const side = index === 0 ? -1 : 1
      const target = targets[index]
      aim.set(side * 1.15, 1.6 + Math.sin(time * 1.4 + index) * 0.04, HIT_Z + 0.45)
      hand.set(side * 0.7, 0.55 + Math.sin(time * 1.7 + index) * 0.025, PLAYER_Z)
      if (target) {
        const untilHit = Math.max(0, (HIT_Z - target.position.z) / NOTE_SPEED)
        const preparation = ease(1 - untilHit / 0.58)
        // A curved backswing accelerates into the cut, with matching speed on
        // the other side of contact instead of snapping to a new pose.
        const offset = -0.9 * Math.sin(Math.min(untilHit / 0.22, 1) * Math.PI / 2)
        pose(target, side, offset)
        aim.lerp(swingAim, preparation)
        hand.lerp(swingHand, preparation)
      }
      const follow = followThrough[index]
      if (follow) {
        follow.age += dt
        const recovery = 1 - ease((follow.age - 0.22) / 0.34)
        const offset = 0.9 * Math.sin(Math.min(follow.age / 0.22, 1) * Math.PI / 2)
        pose(follow.target, side, offset)
        aim.lerp(swingAim, recovery)
        hand.lerp(swingHand, recovery)
        if (follow.age >= 0.56) followThrough[index] = null
      }
      saber.position.lerp(hand, dt === 0 ? 1 : 1 - Math.exp(-16 * dt))
      smoothAims[index].lerp(aim, dt === 0 ? 1 : 1 - Math.exp(-32 * dt))
      rotation.setFromUnitVectors(up, aim.copy(smoothAims[index]).sub(saber.position).normalize())
      saber.quaternion.slerp(rotation, dt === 0 ? 1 : 1 - Math.exp(-32 * dt))
    })
  }
  update(0, 0, [null, null])

  return {
    group,
    update,
    hit(colour: number, target: ReplayTarget) {
      followThrough[colour] = { target: { position: target.position.clone(), direction: target.direction.clone() }, age: 0 }
    },
    dispose() {
      headsetModel.dispose()
      for (const geometry of geometries) geometry.dispose()
      for (const material of materials) material.dispose()
    },
  }
}
