import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/controls/OrbitControls.js';

const canvasStruct = document.getElementById('canvasStruct');
const canvasEmbed = document.getElementById('canvasEmbed');

const viewMode = document.getElementById('viewMode');
const pointSize = document.getElementById('pointSize');
const edgeOpacity = document.getElementById('edgeOpacity');
const showEdges = document.getElementById('showEdges');
const resetBtn = document.getElementById('reset');
const meta = document.getElementById('meta');

const rendererStruct = new THREE.WebGLRenderer({ canvas: canvasStruct, antialias: true });
const rendererEmbed = new THREE.WebGLRenderer({ canvas: canvasEmbed, antialias: true });

rendererStruct.setPixelRatio(window.devicePixelRatio);
rendererEmbed.setPixelRatio(window.devicePixelRatio);

const sceneStruct = new THREE.Scene();
const sceneEmbed = new THREE.Scene();
sceneStruct.fog = new THREE.Fog(0x0b0f1a, 40, 160);
sceneEmbed.fog = new THREE.Fog(0x0b0f1a, 40, 160);

const cameraStruct = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
const cameraEmbed = new THREE.PerspectiveCamera(50, 1, 0.1, 500);

cameraStruct.position.set(0, 0, 80);
cameraEmbed.position.set(0, 0, 80);

const controlsStruct = new OrbitControls(cameraStruct, rendererStruct.domElement);
const controlsEmbed = new OrbitControls(cameraEmbed, rendererEmbed.domElement);
controlsStruct.enableDamping = true;
controlsEmbed.enableDamping = true;

const glowTexture = makeGlowTexture();

let pointsStruct, pointsEmbed, linesStruct, linesEmbed;

function makeGlowTexture() {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(size/2, size/2, 2, size/2, size/2, size/2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.2, 'rgba(255,255,255,0.7)');
  grad.addColorStop(0.6, 'rgba(255,255,255,0.15)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

function colorByHop(hops) {
  const min = Math.min(...hops);
  const max = Math.max(...hops);
  const colors = new Float32Array(hops.length * 3);
  for (let i = 0; i < hops.length; i++) {
    const t = (hops[i] - min) / (max - min + 1e-6);
    const r = 0.2 + 0.7 * t;
    const g = 0.4 + 0.4 * (1 - t);
    const b = 0.9 - 0.7 * t;
    colors[i * 3 + 0] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  return colors;
}

function buildPoints(positions, colors, mutIndex) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: parseFloat(pointSize.value),
    vertexColors: true,
    transparent: true,
    opacity: 0.95,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    map: glowTexture,
  });

  const pts = new THREE.Points(geom, mat);

  // mutation star
  const starGeom = new THREE.SphereGeometry(1.6, 16, 16);
  const starMat = new THREE.MeshBasicMaterial({ color: 0xff3b30 });
  const star = new THREE.Mesh(starGeom, starMat);
  star.position.fromArray(positions, mutIndex * 3);
  pts.add(star);

  return pts;
}

function buildLines(positions, edges) {
  const segments = new Float32Array(edges.length * 2 * 3);
  for (let i = 0; i < edges.length; i++) {
    const [u, v] = edges[i];
    segments.set(positions.slice(u * 3, u * 3 + 3), i * 6);
    segments.set(positions.slice(v * 3, v * 3 + 3), i * 6 + 3);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0xa0a6b8, transparent: true, opacity: parseFloat(edgeOpacity.value) });
  return new THREE.LineSegments(geom, mat);
}

function toFloat32(positions) {
  const arr = new Float32Array(positions.length * 3);
  positions.forEach((p, i) => {
    arr[i * 3 + 0] = p[0];
    arr[i * 3 + 1] = p[1];
    arr[i * 3 + 2] = p[2];
  });
  return arr;
}

function fitCamera(camera, positions) {
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < positions.length; i += 3) {
    v.set(positions[i], positions[i + 1], positions[i + 2]);
    box.expandByPoint(v);
  }
  const size = box.getSize(new THREE.Vector3()).length();
  const center = box.getCenter(new THREE.Vector3());
  camera.position.copy(center.clone().add(new THREE.Vector3(0, 0, size * 0.9 + 30)));
  camera.lookAt(center);
}

async function loadData() {
  const res = await fetch('data.json');
  const data = await res.json();

  meta.textContent = `model: ${data.meta.label} | nodes: ${data.meta.nodes} | edges: ${data.meta.edges} | checkpoint: ${data.meta.checkpoint}`;

  const structPos = toFloat32(data.struct);
  const embedPos = toFloat32(data.embed);
  const colors = colorByHop(data.hop);

  pointsStruct = buildPoints(structPos, colors, data.mut);
  pointsEmbed = buildPoints(embedPos, colors, data.mut);

  linesStruct = buildLines(structPos, data.edges);
  linesEmbed = buildLines(embedPos, data.edges);

  sceneStruct.add(pointsStruct);
  sceneEmbed.add(pointsEmbed);
  sceneStruct.add(linesStruct);
  sceneEmbed.add(linesEmbed);

  fitCamera(cameraStruct, structPos);
  fitCamera(cameraEmbed, embedPos);
}

function resize() {
  const rectStruct = canvasStruct.getBoundingClientRect();
  const rectEmbed = canvasEmbed.getBoundingClientRect();

  rendererStruct.setSize(rectStruct.width, rectStruct.height, false);
  rendererEmbed.setSize(rectEmbed.width, rectEmbed.height, false);

  cameraStruct.aspect = rectStruct.width / rectStruct.height;
  cameraEmbed.aspect = rectEmbed.width / rectEmbed.height;
  cameraStruct.updateProjectionMatrix();
  cameraEmbed.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  controlsStruct.update();
  controlsEmbed.update();

  rendererStruct.render(sceneStruct, cameraStruct);
  rendererEmbed.render(sceneEmbed, cameraEmbed);
}

pointSize.addEventListener('input', () => {
  if (pointsStruct) pointsStruct.material.size = parseFloat(pointSize.value);
  if (pointsEmbed) pointsEmbed.material.size = parseFloat(pointSize.value);
});

edgeOpacity.addEventListener('input', () => {
  if (linesStruct) linesStruct.material.opacity = parseFloat(edgeOpacity.value);
  if (linesEmbed) linesEmbed.material.opacity = parseFloat(edgeOpacity.value);
});

showEdges.addEventListener('change', () => {
  if (linesStruct) linesStruct.visible = showEdges.checked;
  if (linesEmbed) linesEmbed.visible = showEdges.checked;
});

viewMode.addEventListener('change', () => {
  const v = viewMode.value;
  document.querySelectorAll('.panel')[0].style.display = (v === 'embed') ? 'none' : 'block';
  document.querySelectorAll('.panel')[1].style.display = (v === 'struct') ? 'none' : 'block';
  resize();
});

resetBtn.addEventListener('click', () => {
  if (pointsStruct) fitCamera(cameraStruct, pointsStruct.geometry.attributes.position.array);
  if (pointsEmbed) fitCamera(cameraEmbed, pointsEmbed.geometry.attributes.position.array);
});

window.addEventListener('resize', resize);

loadData().then(() => {
  resize();
  animate();
});
