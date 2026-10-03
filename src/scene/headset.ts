import {
  BufferGeometry, CatmullRomCurve3, CylinderGeometry, ExtrudeGeometry, Float32BufferAttribute,
  Group, Mesh, MeshStandardMaterial, Shape, Vector3,
} from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

// Sweep a thin rectangular fabric band, rather than a round tube, along a curve.
function band(points: number[][], width: number, across: Vector3) {
  const curve = new CatmullRomCurve3(points.map((point) => new Vector3(...point)))
  const positions: number[] = []
  const indices: number[] = []
  const steps = 24
  for (let i = 0; i <= steps; i++) {
    const center = curve.getPoint(i / steps)
    const normal = new Vector3().crossVectors(curve.getTangent(i / steps), across).normalize()
    for (const [edge, face] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const vertex = center.clone().addScaledVector(across, edge * width / 2).addScaledVector(normal, face * 0.009)
      positions.push(vertex.x, vertex.y, vertex.z)
    }
    if (i === steps) continue
    for (let j = 0; j < 4; j++) {
      const a = i * 4 + j
      const b = i * 4 + (j + 1) % 4
      indices.push(a, b, a + 4, b, b + 4, a + 4)
    }
  }
  indices.push(0, 2, 1, 0, 3, 2)
  const end = steps * 4
  indices.push(end, end + 1, end + 2, end, end + 2, end + 3)
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  return geometry
}

export function createHeadset() {
  const group = new Group()
  const plastic = new MeshStandardMaterial({ color: 0xf0f0ed, roughness: 0.5, metalness: 0, emissive: 0x181818, emissiveIntensity: 0.2 })
  const fabric = new MeshStandardMaterial({ color: 0xc9c8c4, roughness: 0.95 })
  const foam = new MeshStandardMaterial({ color: 0x202129, roughness: 1 })
  const sensor = new MeshStandardMaterial({ color: 0x080b13, roughness: 0.23, metalness: 0.15 })
  const lens = new MeshStandardMaterial({ color: 0x142335, roughness: 0.08, metalness: 0.65 })
  const materials = [plastic, fabric, foam, sensor, lens]
  const add = (geometry: BufferGeometry, material: MeshStandardMaterial, x = 0, y = 0, z = 0) => {
    const mesh = new Mesh(geometry, material)
    mesh.position.set(x, y, z)
    group.add(mesh)
    return mesh
  }
  const curvedBox = (width: number, height: number, depth: number, radius: number) => {
    const geometry = new RoundedBoxGeometry(width, height, depth, 4, radius)
    const positions = geometry.attributes.position
    const normals = geometry.attributes.normal
    const normal = new Vector3()
    for (let i = 0; i < positions.count; i++) {
      // Wrap the shell gently around the face instead of using a flat brick.
      const x = positions.getX(i)
      positions.setZ(i, positions.getZ(i) + 0.075 * (x / 0.5) ** 2)
      // Preserve the rounded box's smooth normals while bending the surface.
      normal.fromBufferAttribute(normals, i)
      normal.x -= 0.6 * x * normal.z
      normal.normalize()
      normals.setXYZ(i, normal.x, normal.y, normal.z)
    }
    return geometry
  }

  add(curvedBox(0.88, 0.38, 0.16, 0.075), foam, 0, -0.015, 0.09)
  add(curvedBox(1, 0.43, 0.24, 0.095), plastic, 0, 0.01, -0.075)

  // Three recessed vertical sensor windows on the front of the visor.
  for (const x of [-0.31, 0, 0.31]) {
    const z = -0.202 + 0.075 * (x / 0.5) ** 2
    const pill = new Shape()
    pill.absarc(0, 0.0775, 0.045, 0, Math.PI, false)
    pill.lineTo(-0.045, -0.0775)
    pill.absarc(0, -0.0775, 0.045, Math.PI, Math.PI * 2, false)
    pill.closePath()
    const panelGeometry = new ExtrudeGeometry(pill, { depth: 0.01, bevelEnabled: true, bevelThickness: 0.002, bevelSize: 0.002, bevelSegments: 2, curveSegments: 10, steps: 1 })
    panelGeometry.translate(0, 0, -0.005)
    const panel = add(panelGeometry, sensor, x, 0.015, z)
    panel.rotation.y = -x * 0.6
    for (const y of [-0.054, 0.077]) {
      const camera = add(new CylinderGeometry(0.025, 0.025, 0.008, 12), lens, x, y, z - 0.011)
      camera.rotation.x = Math.PI / 2
    }
  }
  // Eye cups make the open back read as a headset even from the replay camera.
  for (const x of [-0.21, 0.21]) {
    const cup = add(new CylinderGeometry(0.143, 0.16, 0.055, 20), sensor, x, -0.015, 0.185)
    cup.rotation.x = Math.PI / 2
    cup.scale.z = 0.86
    const eye = add(new CylinderGeometry(0.119, 0.119, 0.009, 20), lens, x, -0.015, 0.216)
    eye.rotation.x = Math.PI / 2
    eye.scale.z = 0.86
  }
  for (const side of [-1, 1]) {
    const arm = add(new RoundedBoxGeometry(0.085, 0.11, 0.28, 3, 0.03), plastic, side * 0.465, 0.005, 0.15)
    arm.rotation.y = side * -0.1
    // Split crown bands join the broad rear strap, leaving no rigid halo.
    add(band([[side * 0.06, 0.21, -0.045], [side * 0.08, 0.34, 0.18], [side * 0.14, 0.27, 0.41], [side * 0.17, 0.07, 0.57]], 0.075, new Vector3(1, 0, 0)), fabric)
  }
  add(band([[-0.46, 0.01, 0.2], [-0.43, 0.015, 0.39], [-0.27, 0.02, 0.55], [0, 0.02, 0.6], [0.27, 0.02, 0.55], [0.43, 0.015, 0.39], [0.46, 0.01, 0.2]], 0.1, new Vector3(0, 1, 0)), fabric)
  add(new RoundedBoxGeometry(0.25, 0.13, 0.027, 2, 0.025), fabric, 0, 0.02, 0.603)

  // Keep the extra modeling detail to five draw calls.
  const batches = new Map<MeshStandardMaterial, BufferGeometry[]>()
  for (const child of [...group.children]) {
    const mesh = child as Mesh<BufferGeometry, MeshStandardMaterial>
    mesh.updateMatrix()
    const geometry = mesh.geometry.index ? mesh.geometry.toNonIndexed() : mesh.geometry.clone()
    geometry.applyMatrix4(mesh.matrix)
    geometry.deleteAttribute('uv')
    const batch = batches.get(mesh.material) ?? []
    batch.push(geometry)
    batches.set(mesh.material, batch)
    mesh.geometry.dispose()
    group.remove(mesh)
  }
  const geometries = [...batches].map(([material, parts]) => {
    const geometry = mergeGeometries(parts)!
    for (const part of parts) part.dispose()
    group.add(new Mesh(geometry, material))
    return geometry
  })
  return {
    group,
    dispose() {
      for (const geometry of geometries) geometry.dispose()
      for (const material of materials) material.dispose()
    },
  }
}
