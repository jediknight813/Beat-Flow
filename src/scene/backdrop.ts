import {
  AdditiveBlending,
  AmbientLight,
  BufferAttribute,
  BufferGeometry,
  Timer,
  Color,
  DirectionalLight,
  DoubleSide,
  Fog,
  Group,
  HalfFloatType,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  NeutralToneMapping,
  PerspectiveCamera,
  Plane,
  PlaneGeometry,
  Points,
  PointsMaterial,
  Quaternion,
  Raycaster,
  Scene,
  Shape,
  ShapeGeometry,
  Vector2,
  Vector3,
  WebGLRenderer,
  WebGLRenderTarget,
} from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { faceTexture, sparkTexture } from './faces'
import { createArena } from './arena'
import { HIT_Z, LANE_CENTERS, NOTE_SPEED } from './layout'
import { createPlayer, type ReplayTarget } from './player'

const SIZE = 0.72
const RADIUS = 0.035
const COUNT = 24
const HALF_LIFE = 0.6
const HEAT_LIFE = 0.3
const FLASH_LIFE = 0.18
const BURST_LIFE = 0.5
const Z = new Vector3(0, 0, 1)
const WHITE = new Color(1, 1, 1)
const HOT = new Color(3, 3, 3)

const arrows = [
  [0, 1],
  [0, -1],
  [-1, 0],
  [1, 0],
  [-1, 1],
  [1, 1],
  [-1, -1],
  [1, -1],
].map(([x, y]) => new Vector3(x, y, 0).normalize())


type Block = {
  mesh: Mesh<RoundedBoxGeometry, MeshStandardMaterial>
  face: Mesh<PlaneGeometry, MeshBasicMaterial>
  colour: number
  direction: number
  phase: number
  home: Vector3
  tilt: Vector3
}

type Effect = { update(dt: number): boolean; dispose(): void }

const rand = (min: number, max: number) => min + Math.random() * (max - min)

function roundedRect(width: number, height: number, radius: number) {
  const x = -width / 2
  const y = -height / 2
  const shape = new Shape()
  shape.moveTo(x + radius, y)
  shape.lineTo(x + width - radius, y)
  shape.quadraticCurveTo(x + width, y, x + width, y + radius)
  shape.lineTo(x + width, y + height - radius)
  shape.quadraticCurveTo(x + width, y + height, x + width - radius, y + height)
  shape.lineTo(x + radius, y + height)
  shape.quadraticCurveTo(x, y + height, x, y + height - radius)
  shape.lineTo(x, y + radius)
  shape.quadraticCurveTo(x, y, x + radius, y)
  return new ShapeGeometry(shape)
}

function step(effects: Effect[], dt: number) {
  let count = 0
  for (const effect of effects) {
    if (effect.update(dt)) effects[count++] = effect
    else effect.dispose()
  }
  effects.length = count
  return effects
}

export function createBackdrop(canvas: HTMLCanvasElement) {
  const motionPreference = matchMedia('(prefers-reduced-motion: reduce)')
  let reducedMotion = motionPreference.matches

  // StrictMode and hot reload can reuse this canvas's WebGL context. Three's
  // placeholder 3D textures require these unpack flags to be reset first.
  const context = canvas.getContext('webgl2', { antialias: false, powerPreference: 'low-power' })
  if (!context) return () => {}
  context.pixelStorei(context.UNPACK_FLIP_Y_WEBGL, false)
  context.pixelStorei(context.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false)
  const renderer = new WebGLRenderer({ canvas, context, antialias: false })
  renderer.toneMapping = NeutralToneMapping
  renderer.localClippingEnabled = true

  const scene = new Scene()
  scene.background = new Color(0x070611)
  const fog = new Fog(0x070611, 14, 65)
  scene.fog = fog

  const camera = new PerspectiveCamera(58, 1, 0.1, 100)
  const cameraTarget = new Vector3(0, 0.5, -1.5)
  camera.position.set(-8, 13, 13.1)
  camera.lookAt(cameraTarget)
  camera.updateMatrixWorld()

  const composer = new EffectComposer(renderer, new WebGLRenderTarget(1, 1, { type: HalfFloatType, samples: 4 }))
  const bloom = new UnrealBloomPass(new Vector2(1, 1), 0.3, 0.35, 1.15)
  const passes = [new RenderPass(scene, camera), bloom, new OutputPass()]
  for (const pass of passes) composer.addPass(pass)

  const ambient = new AmbientLight(0x9aabdd, 0.65)
  const key = new DirectionalLight(0xffffff, 2.5)
  key.position.set(3, 7, 5)
  const rim = new DirectionalLight(0x4488ff, 1.2)
  rim.position.set(-5, 2, -9)
  const edge = new DirectionalLight(0xff2244, 1.2)
  edge.position.set(6, 4, -6)
  scene.add(ambient, key, rim, edge)
  const arena = createArena()
  scene.add(arena.group)
  const player = createPlayer()
  scene.add(player.group)

  const colours = [new Color(0xe60025), new Color(0x008bcc)]
  const bodies = colours.map((colour) => new MeshStandardMaterial({ color: colour, emissive: colour, emissiveIntensity: 0.12, roughness: 0.25, metalness: 0.45 }))
  const caps = colours.map((colour) => new MeshStandardMaterial({ color: colour, emissive: WHITE, emissiveIntensity: 3, roughness: 0.5, side: DoubleSide }))
  const textures = Array.from({ length: 9 }, (_, direction) => faceTexture(direction))
  const faces = textures.map((map) => new MeshBasicMaterial({ map, color: new Color(1.15, 1.15, 1.15), transparent: true, toneMapped: false }))
  const spark = sparkTexture()

  const box = new RoundedBoxGeometry(SIZE, SIZE, SIZE, 2, RADIUS)
  const facePlane = new PlaneGeometry(0.63, 0.63)
  const capStraight = roundedRect(SIZE, SIZE, RADIUS)
  const capDiagonal = roundedRect(SIZE * Math.SQRT2, SIZE, RADIUS)
  const flashPlane = new PlaneGeometry(1, 1)

  const blocks: Block[] = []
  for (let i = 0; i < COUNT; i++) {
    const colour = i % 2
    const mesh = new Mesh(box, bodies[colour])
    const face = new Mesh(facePlane, faces[0])
    face.position.z = SIZE / 2 + 0.004
    mesh.add(face)
    scene.add(mesh)
    blocks.push({ mesh, face, colour, direction: 0, phase: 0, home: new Vector3(), tilt: new Vector3() })
  }

  const place = (block: Block, initial: boolean) => {
    const index = blocks.indexOf(block)
    const pair = Math.floor(index / 2)
    const lane = block.colour === 0 ? pair % 2 : 3 - pair % 2
    block.home.set(LANE_CENTERS[lane], 0.5 + (index % 3) * 0.65,
      initial ? 1 - Math.floor(index / 2) * 4.5 : -51 - rand(0, 3))
  }

  const spawn = (block: Block, initial: boolean) => {
    block.direction = Math.floor(Math.random() * 9)
    block.face.material = faces[block.direction]
    block.phase = rand(0, Math.PI * 2)
    block.tilt.set(rand(-0.08, 0.08), -0.12, rand(-0.14, 0.14))
    place(block, initial)
    block.mesh.position.copy(block.home)
    block.mesh.rotation.set(block.tilt.x, block.tilt.y, block.tilt.z)
    block.mesh.visible = true
    dirty = true
  }

  const half = (block: Block, origin: Vector3, rotation: Quaternion, along: Vector3, normal: Vector3, sign: number): Effect => {
    const keep = normal.clone().multiplyScalar(sign)
    const colour = colours[block.colour]
    const plane = new Plane()
    const body = new Mesh(box, bodies[block.colour].clone())
    const face = new Mesh(facePlane, faces[block.direction].clone())
    const cap = new Mesh(block.direction < 4 ? capStraight : capDiagonal, caps[block.colour].clone())
    face.position.z = SIZE / 2 + 0.004
    cap.quaternion.setFromRotationMatrix(new Matrix4().makeBasis(along, Z, normal.clone().negate()))
    body.material.clippingPlanes = [plane]
    face.material.clippingPlanes = [plane]
    const materials = [body.material, face.material, cap.material]
    for (const material of materials) material.transparent = true
    const group = new Group().add(body, face, cap)
    group.position.copy(origin)
    group.quaternion.copy(rotation)
    scene.add(group)
    const velocity = keep.clone().applyQuaternion(rotation).multiplyScalar(1.3).add(new Vector3(rand(-0.3, 0.3), 0.4, 0.8))
    const axis = along.clone().applyQuaternion(rotation).multiplyScalar(sign).add(new Vector3(rand(-0.3, 0.3), rand(-0.3, 0.3), rand(-0.3, 0.3))).normalize()
    const rate = rand(2.5, 4)
    const turn = new Quaternion()
    let t = 0
    return {
      update(dt) {
        t += dt
        velocity.y -= 4 * dt
        group.position.addScaledVector(velocity, dt)
        group.quaternion.premultiply(turn.setFromAxisAngle(axis, rate * dt))
        group.updateMatrixWorld()
        plane.set(keep, 0).applyMatrix4(group.matrixWorld)
        const heat = 1 - Math.min(t / HEAT_LIFE, 1)
        cap.material.emissive.copy(colour).lerp(WHITE, heat)
        cap.material.emissiveIntensity = 0.6 + 2.4 * heat
        const k = Math.min(t / HALF_LIFE, 1)
        for (const material of materials) material.opacity = 1 - k * k
        return t < HALF_LIFE
      },
      dispose() {
        scene.remove(group)
        for (const material of materials) material.dispose()
      },
    }
  }

  const flash = (origin: Vector3, rotation: Quaternion, along: Vector3, normal: Vector3): Effect => {
    const material = new MeshBasicMaterial({
      color: HOT,
      transparent: true,
      blending: AdditiveBlending,
      side: DoubleSide,
      depthWrite: false,
      toneMapped: false,
    })
    const mesh = new Mesh(flashPlane, material)
    mesh.position.copy(origin)
    mesh.quaternion.copy(rotation).multiply(new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(along, Z, normal.clone().negate())))
    scene.add(mesh)
    let t = 0
    return {
      update(dt) {
        t += dt
        const k = Math.min(t / FLASH_LIFE, 1)
        mesh.scale.setScalar(0.62 + k * 0.3)
        material.opacity = 1 - k
        return t < FLASH_LIFE
      },
      dispose() {
        scene.remove(mesh)
        material.dispose()
      },
    }
  }

  const burst = (origin: Vector3, colour: Color): Effect => {
    const count = 20
    const positions = new Float32Array(count * 3)
    const velocities = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      const v = new Vector3().randomDirection().multiplyScalar(rand(0.6, 2.2))
      positions.set([origin.x, origin.y, origin.z], i * 3)
      velocities.set([v.x, v.y, v.z + 0.6], i * 3)
    }
    const attribute = new BufferAttribute(positions, 3)
    const geometry = new BufferGeometry().setAttribute('position', attribute)
    const material = new PointsMaterial({
      map: spark,
      color: colour.clone().lerp(WHITE, 0.5),
      size: 0.18,
      transparent: true,
      blending: AdditiveBlending,
      depthWrite: false,
    })
    const points = new Points(geometry, material)
    points.frustumCulled = false
    scene.add(points)
    let t = 0
    return {
      update(dt) {
        t += dt
        for (let i = 0; i < positions.length; i += 3) {
          velocities[i + 1] -= 2.5 * dt
          positions[i] += velocities[i] * dt
          positions[i + 1] += velocities[i + 1] * dt
          positions[i + 2] += velocities[i + 2] * dt
        }
        attribute.needsUpdate = true
        material.opacity = 1 - t / BURST_LIFE
        return t < BURST_LIFE
      },
      dispose() {
        scene.remove(points)
        geometry.dispose()
        material.dispose()
      },
    }
  }

  const slice = (block: Block): Effect => {
    const along = arrows[block.direction === 8 ? Math.floor(Math.random() * 8) : block.direction]
    const normal = new Vector3(-along.y, along.x, 0)
    const origin = block.mesh.position.clone()
    const rotation = block.mesh.quaternion.clone()
    const colour = colours[block.colour]
    block.mesh.visible = false
    let parts = [
      half(block, origin, rotation, along, normal, 1),
      half(block, origin, rotation, along, normal, -1),
      flash(origin, rotation, along, normal),
      burst(origin, colour),
    ]
    return {
      update(dt) {
        parts = step(parts, dt)
        if (parts.length) return true
        spawn(block, false)
        return false
      },
      dispose() {
        for (const part of parts) part.dispose()
      },
    }
  }

  let effects: Effect[] = []
  let dirty = true
  let frame = 0
  let lastFrameAt = 0
  let time = 0
  let hoverAt = 0
  let pointerInside = false
  let gesture: { id: number; x: number; y: number; dragging: boolean } | null = null
  const pointer = new Vector2()
  const raycaster = new Raycaster()
  const timer = new Timer()
  timer.connect(document)
  const targets: (ReplayTarget | null)[] = [null, null]
  const blockTargets = blocks.map((block) => ({ position: block.mesh.position, direction: arrows[0] }))
  const pickMeshes: Block['mesh'][] = []

  const pick = () => {
    raycaster.setFromCamera(pointer, camera)
    pickMeshes.length = 0
    for (const block of blocks) if (block.mesh.visible) pickMeshes.push(block.mesh)
    const hit = raycaster.intersectObjects(pickMeshes, false)[0]
    return hit && blocks.find((block) => block.mesh === hit.object)
  }

  const requestFrame = () => {
    if (frame || document.hidden) return
    timer.reset()
    frame = requestAnimationFrame(tick)
  }

  const resize = () => {
    // Bound full-screen bloom and multisample buffers on large/HiDPI displays.
    const pixelRatio = Math.min(devicePixelRatio, 1.5, Math.sqrt(2_000_000 / (innerWidth * innerHeight)))
    renderer.setPixelRatio(pixelRatio)
    composer.setPixelRatio(pixelRatio)
    renderer.setSize(innerWidth, innerHeight, false)
    composer.setSize(innerWidth, innerHeight)
    camera.aspect = innerWidth / innerHeight
    // Fill the extra desktop space while keeping the locked viewing angle.
    // Wider displays bring the track closer and slightly toward the controls.
    const largeScreen = Math.min(1, Math.max(0, (innerWidth - 1440) / 1120))
    const wideScreen = Math.min(1, Math.max(0, (camera.aspect - 16 / 9) / (5 / 9)))
    camera.zoom = 1 + largeScreen * 0.3 + wideScreen * 0.15
    const horizontalOffset = innerWidth > 600 ? 0.19 - largeScreen * 0.05 - wideScreen * 0.02 : 0
    camera.setViewOffset(innerWidth, innerHeight, -innerWidth * horizontalOffset, 0, innerWidth, innerHeight)
    camera.updateProjectionMatrix()
    dirty = true
    requestFrame()
  }

  const onPointerMove = (event: PointerEvent) => {
    pointer.set((event.clientX / innerWidth) * 2 - 1, 1 - (event.clientY / innerHeight) * 2)
    pointerInside = true
    if (reducedMotion && !gesture) canvas.style.cursor = pick() ? 'pointer' : ''
    if (!gesture || gesture.id !== event.pointerId) return
    const dx = event.clientX - gesture.x
    const dy = event.clientY - gesture.y
    if (Math.hypot(dx, dy) > 5) gesture.dragging = true
  }

  const onPointerLeave = () => {
    pointerInside = false
    if (!gesture) canvas.style.cursor = ''
  }

  const onPointerDown = (event: PointerEvent) => {
    if (event.button !== 0 || gesture) return
    onPointerMove(event)
    gesture = { id: event.pointerId, x: event.clientX, y: event.clientY, dragging: false }
    canvas.setPointerCapture(event.pointerId)
  }

  const endGesture = () => {
    const id = gesture?.id
    gesture = null
    if (id !== undefined && canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id)
    canvas.style.cursor = ''
  }

  const onPointerUp = (event: PointerEvent) => {
    if (gesture?.id !== event.pointerId) return
    onPointerMove(event)
    const dragged = gesture.dragging
    endGesture()
    if (!dragged) {
      const block = pick()
      if (block) {
        effects.push(slice(block))
        dirty = true
        requestFrame()
      }
    }
  }

  const tick = (timestamp: number) => {
    frame = 0
    // A decorative scene does not need to render at 120/144 Hz.
    if (timestamp - lastFrameAt < 1000 / 60 - 0.5) {
      frame = requestAnimationFrame(tick)
      return
    }
    lastFrameAt = timestamp
    timer.update(timestamp)
    const dt = Math.min(timer.getDelta(), 0.05)
    time += dt
    if (!reducedMotion) {
      targets.fill(null)
      for (let index = 0; index < blocks.length; index++) {
        const block = blocks[index]
        if (!block.mesh.visible) continue
        block.home.z += NOTE_SPEED * dt
        block.mesh.position.set(block.home.x, block.home.y + Math.sin(time * 0.7 + block.phase) * 0.015, block.home.z)
        block.mesh.rotation.set(
          block.tilt.x + Math.sin(time * 0.5 + block.phase) * 0.05,
          block.tilt.y + Math.sin(time * 0.4 + block.phase * 1.3) * 0.05,
          block.tilt.z,
        )
        const target = blockTargets[index]
        target.direction = arrows[block.direction === 8 ? 1 : block.direction]
        if (block.home.z >= HIT_Z) {
          player.hit(block.colour, target)
          effects.push(slice(block))
        } else if (!targets[block.colour] || block.home.z > targets[block.colour]!.position.z) {
          targets[block.colour] = target
        }
      }
      player.update(time, dt, targets)
    }
    effects = step(effects, dt)
    if (pointerInside && !gesture && time - hoverAt > 0.08) {
      hoverAt = time
      canvas.style.cursor = pick() ? 'pointer' : ''
    }
    if (!reducedMotion || effects.length || dirty) composer.render(dt)
    dirty = false
    if (!reducedMotion || effects.length) frame = requestAnimationFrame(tick)
  }

  const onMotionChange = () => {
    reducedMotion = motionPreference.matches
    dirty = true
    requestFrame()
  }

  const onVisibility = () => {
    if (document.hidden) {
      endGesture()
      cancelAnimationFrame(frame)
      frame = 0
    } else requestFrame()
  }

  for (const block of blocks) spawn(block, true)
  resize()

  addEventListener('resize', resize)
  motionPreference.addEventListener('change', onMotionChange)
  document.addEventListener('visibilitychange', onVisibility)
  canvas.addEventListener('pointermove', onPointerMove)
  canvas.addEventListener('pointerleave', onPointerLeave)
  canvas.addEventListener('pointerdown', onPointerDown)
  canvas.addEventListener('pointerup', onPointerUp)
  canvas.addEventListener('pointercancel', endGesture)
  canvas.addEventListener('lostpointercapture', endGesture)
  onVisibility()

  return () => {
    endGesture()
    cancelAnimationFrame(frame)
    removeEventListener('resize', resize)
    motionPreference.removeEventListener('change', onMotionChange)
    document.removeEventListener('visibilitychange', onVisibility)
    canvas.removeEventListener('pointermove', onPointerMove)
    canvas.removeEventListener('pointerleave', onPointerLeave)
    canvas.removeEventListener('pointerdown', onPointerDown)
    canvas.removeEventListener('pointerup', onPointerUp)
    canvas.removeEventListener('pointercancel', endGesture)
    canvas.removeEventListener('lostpointercapture', endGesture)
    for (const effect of effects) effect.dispose()
    for (const pass of passes) pass.dispose()
    composer.dispose()
    timer.dispose()
    for (const geometry of [box, facePlane, capStraight, capDiagonal, flashPlane]) geometry.dispose()
    for (const material of [...bodies, ...caps, ...faces]) material.dispose()
    for (const texture of [...textures, spark]) texture.dispose()
    arena.dispose()
    player.dispose()
    renderer.resetState()
    renderer.dispose()
  }
}
