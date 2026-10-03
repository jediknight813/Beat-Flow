import {
  AdditiveBlending, BoxGeometry, Color, Group, Mesh, MeshBasicMaterial,
  MeshStandardMaterial, PointLight, Sprite, SpriteMaterial, Vector3,
} from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { sparkTexture } from './faces'
import { LANE_CENTERS, PORTAL_X, RAIL_X, TRACK_HALF_WIDTH } from './layout'

// Architecture and light bars share the same perspective as the note lanes.
export function createArena() {
  const group = new Group()
  const cube = new BoxGeometry(1, 1, 1)
  const steel = new MeshStandardMaterial({ color: 0x10101c, roughness: 0.72, metalness: 0.5 })
  const floor = new MeshStandardMaterial({ color: 0x090b16, roughness: 0.32, metalness: 0.65 })
  const red = new MeshBasicMaterial({ color: new Color(3.5, 0.025, 0.12), toneMapped: false })
  const blue = new MeshBasicMaterial({ color: new Color(0.025, 1.6, 3), toneMapped: false })
  const dim = new MeshBasicMaterial({ color: 0x25263f })
  const bar = (a: number[], b: number[], width: number, material: MeshBasicMaterial | MeshStandardMaterial) => {
    const start = new Vector3(...a)
    const end = new Vector3(...b)
    const mesh = new Mesh(cube, material)
    mesh.position.copy(start).add(end).multiplyScalar(0.5)
    mesh.scale.set(width, start.distanceTo(end), width)
    mesh.quaternion.setFromUnitVectors(new Vector3(0, 1, 0), end.sub(start).normalize())
    group.add(mesh)
  }
  const platform = new Mesh(cube, floor)
  platform.position.set(0, -1.15, -27)
  platform.scale.set(TRACK_HALF_WIDTH * 2, 0.3, 72)
  group.add(platform)

  for (const x of LANE_CENTERS) {
    bar([x, -0.97, 9], [x, -0.97, -63], 0.012, dim)
  }

  for (let i = 0; i < 12; i++) {
    const z = 3 - i * 6
    // Each light is attached to the inward face of its structural upright.
    for (const side of [-1, 1]) {
      bar([side * PORTAL_X, -2, z], [side * PORTAL_X, 6, z], 0.14, steel)
      bar([side * PORTAL_X, 6, z], [0, 7, z], 0.14, steel)
      bar([side * RAIL_X, -0.8, z], [side * RAIL_X, 0.4, z], 0.06, steel)
      bar([side * RAIL_X, 0.4, z], [side * RAIL_X, 0.4, z - 6], 0.04, dim)
      const lightX = side * (PORTAL_X - 0.085)
      bar([lightX, 0.8, z], [lightX, 5.6, z], 0.065, side < 0 ? red : blue)
      bar([side * TRACK_HALF_WIDTH, -0.96, z], [side * TRACK_HALF_WIDTH, -0.96, z - 6], 0.035, side < 0 ? red : blue)
    }
    bar([-TRACK_HALF_WIDTH, -0.97, z], [TRACK_HALF_WIDTH, -0.97, z], 0.022, dim)
  }
  // The arena never moves: batch its hundreds of bars into one draw per material.
  const batches = new Map<MeshBasicMaterial | MeshStandardMaterial, BoxGeometry[]>()
  for (const object of [...group.children]) {
    if (!(object instanceof Mesh)) continue
    object.updateMatrix()
    const geometry = object.geometry.clone().applyMatrix4(object.matrix) as BoxGeometry
    const material = object.material as MeshBasicMaterial | MeshStandardMaterial
    const batch = batches.get(material) ?? []
    batch.push(geometry)
    batches.set(material, batch)
    group.remove(object)
  }
  const mergedGeometries = [...batches].map(([material, geometries]) => {
    const merged = mergeGeometries(geometries)!
    for (const geometry of geometries) geometry.dispose()
    const mesh = new Mesh(merged, material)
    mesh.matrixAutoUpdate = false
    group.add(mesh)
    return merged
  })
  // Composite soft atmosphere over the scene. Depth-testing a camera-facing
  // haze sprite cuts it against the floor and produces hard rectangular edges.
  const haze = sparkTexture()
  for (const [x, y, z, color, size] of [
    [-8, 4, -24, 0xe60038, 32], [9, 3, -30, 0x006eff, 35],
    [0, 4, -48, 0x791157, 26],
  ]) {
    const material = new SpriteMaterial({
      map: haze, color, transparent: true, opacity: 0.12,
      blending: AdditiveBlending, depthWrite: false, depthTest: false, fog: false,
    })
    const sprite = new Sprite(material)
    sprite.position.set(x, y, z)
    sprite.scale.set(size, size, 1)
    sprite.renderOrder = 1
    group.add(sprite)
  }
  const leftLight = new PointLight(0xff123e, 65, 35, 2)
  leftLight.position.set(-4, 3, -5)
  const rightLight = new PointLight(0x008dff, 85, 35, 2)
  rightLight.position.set(5, 4, -7)
  group.add(leftLight, rightLight)

  return {
    group,
    dispose() {
      cube.dispose()
      for (const geometry of mergedGeometries) geometry.dispose()
      for (const material of [steel, floor, red, blue, dim]) material.dispose()
      group.traverse((object) => { if (object instanceof Sprite) object.material.dispose() })
      haze.dispose()
    },
  }
}
