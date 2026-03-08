const meta = document.getElementById('meta');
const viewMode = document.getElementById('viewMode');
const pointSize = document.getElementById('pointSize');
const edgeOpacity = document.getElementById('edgeOpacity');
const showEdges = document.getElementById('showEdges');
const showAnn = document.getElementById('showAnn');
const resetBtn = document.getElementById('reset');
const presentBtn = document.getElementById('present');

const panelStruct = document.getElementById('panelStruct');
const panelEmbed = document.getElementById('panelEmbed');
const plotStruct = document.getElementById('plotStruct');
const plotEmbed = document.getElementById('plotEmbed');

let cachedData = null;
let cameraStruct = null;
let cameraEmbed = null;
let presenting = false;
let orbitTimer = null;

const STRUCT_SCALE = [
  [0.0, '#2dd4ff'],
  [0.5, '#ffd166'],
  [1.0, '#ff6b6b'],
];

const EMBED_SCALE = [
  [0.0, '#7cf4a4'],
  [0.5, '#ffb86c'],
  [1.0, '#ff5c5c'],
];

const DEFAULT_CAMERA = {
  eye: { x: 1.15, y: 1.1, z: 0.85 },
  up: { x: 0, y: 0, z: 1 },
  center: { x: 0, y: 0, z: 0 },
};

function normalize(arr) {
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  return arr.map(v => (v - min) / (max - min + 1e-6));
}

function buildEdges(points, edges, opacity) {
  const x = [], y = [], z = [];
  for (const [u, v] of edges) {
    x.push(points[u][0], points[v][0], null);
    y.push(points[u][1], points[v][1], null);
    z.push(points[u][2], points[v][2], null);
  }
  return {
    type: 'scatter3d',
    mode: 'lines',
    x, y, z,
    line: { color: `rgba(170,190,220,${opacity})`, width: 2.4 },
    hoverinfo: 'skip',
    showlegend: false,
  };
}

function buildPoints(points, colors, size, colorscale, showScale, colorbarTitle, labels) {
  return {
    type: 'scatter3d',
    mode: 'markers',
    x: points.map(p => p[0]),
    y: points.map(p => p[1]),
    z: points.map(p => p[2]),
    text: labels,
    hoverinfo: 'text',
    marker: {
      size,
      color: colors,
      colorscale,
      opacity: 0.98,
      line: { color: 'rgba(10,15,31,0.3)', width: 0.5 },
      showscale: showScale,
      colorbar: showScale ? {
        title: { text: colorbarTitle, font: { color: '#e9efff', size: 11 } },
        tickfont: { color: '#cbd5f5', size: 10 },
        len: 0.6,
        outlinewidth: 0,
      } : undefined,
    },
    showlegend: false,
  };
}

function buildNeighborhood(points, hopRaw, maxHop, color, sizeBoost) {
  const idx = [];
  for (let i = 0; i < hopRaw.length; i += 1) {
    if (hopRaw[i] <= maxHop) idx.push(i);
  }
  if (!idx.length) return null;
  return {
    type: 'scatter3d',
    mode: 'markers',
    x: idx.map(i => points[i][0]),
    y: idx.map(i => points[i][1]),
    z: idx.map(i => points[i][2]),
    marker: {
      size: sizeBoost,
      color,
      opacity: 0.9,
      line: { color: 'rgba(255,255,255,0.6)', width: 0.6 },
    },
    hoverinfo: 'skip',
    showlegend: false,
  };
}

function buildMut(points, idx) {
  const p = points[idx];
  const glow = {
    type: 'scatter3d',
    mode: 'markers',
    x: [p[0]], y: [p[1]], z: [p[2]],
    marker: { size: 22, color: 'rgba(255,80,80,0.25)' },
    hoverinfo: 'skip',
    showlegend: false,
  };
  const core = {
    type: 'scatter3d',
    mode: 'markers',
    x: [p[0]], y: [p[1]], z: [p[2]],
    marker: { size: 11, color: '#ff3b3b' },
    hoverinfo: 'skip',
    showlegend: false,
  };
  return [glow, core];
}

function buildAnn(points, ann) {
  const idx = ann.indices || [];
  return {
    type: 'scatter3d',
    mode: 'markers',
    x: idx.map(i => points[i][0]),
    y: idx.map(i => points[i][1]),
    z: idx.map(i => points[i][2]),
    marker: { size: 7, color: ann.color || '#ff4d4d', opacity: 0.95 },
    name: ann.name || 'annotation',
    showlegend: false,
  };
}

function layout(title, camera) {
  return {
    title: { text: title, font: { color: '#ffe9b6', size: 14 } },
    paper_bgcolor: '#0f1629',
    plot_bgcolor: '#0f1629',
    margin: { l: 0, r: 0, t: 30, b: 0 },
    scene: {
      bgcolor: '#0f1629',
      xaxis: { visible: false },
      yaxis: { visible: false },
      zaxis: { visible: false },
      camera: camera || DEFAULT_CAMERA,
      aspectmode: 'data',
    },
  };
}

function renderPlots() {
  const data = cachedData;
  const hopRaw = data.hop;
  const hop = normalize(hopRaw);
  const size = parseInt(pointSize.value, 10);
  const eOpacity = parseFloat(edgeOpacity.value);
  const edges = data.edges;

  const labels = hopRaw.map((h, i) => `Residue ${i} · hop ${h}`);

  const mutStruct = buildMut(data.struct, data.mut);
  const mutEmbed = buildMut(data.embed, data.mut);

  const neighStruct = buildNeighborhood(data.struct, hopRaw, 2, '#7cf4ff', size + 2);
  const neighEmbed = buildNeighborhood(data.embed, hopRaw, 2, '#7cf4a4', size + 2);

  const tracesStruct = [
    buildPoints(data.struct, hop, size, STRUCT_SCALE, false, '', labels),
    ...(neighStruct ? [neighStruct] : []),
    ...mutStruct,
  ];
  const tracesEmbed = [
    buildPoints(data.embed, hop, size, EMBED_SCALE, true, 'Graph hop from mutation', labels),
    ...(neighEmbed ? [neighEmbed] : []),
    ...mutEmbed,
  ];

  if (showEdges.checked) {
    tracesStruct.push(buildEdges(data.struct, edges, eOpacity));
    tracesEmbed.push(buildEdges(data.embed, edges, eOpacity));
  }

  if (showAnn.checked && data.annotations && data.annotations.length) {
    for (const ann of data.annotations) {
      tracesStruct.push(buildAnn(data.struct, ann));
      tracesEmbed.push(buildAnn(data.embed, ann));
    }
  }

  const layoutStruct = layout('Structure Graph', cameraStruct);
  const layoutEmbed = layout('Embedding Space (PCA‑3D)', cameraEmbed);

  Plotly.react(plotStruct, tracesStruct, layoutStruct, {
    displayModeBar: true,
    responsive: true,
  });
  Plotly.react(plotEmbed, tracesEmbed, layoutEmbed, {
    displayModeBar: true,
    responsive: true,
  });
}

function startOrbit() {
  stopOrbit();
  let angle = 0;
  orbitTimer = setInterval(() => {
    angle += 0.02;
    const cam = {
      eye: { x: 1.35 * Math.cos(angle), y: 1.35 * Math.sin(angle), z: 0.95 },
      up: { x: 0, y: 0, z: 1 },
      center: { x: 0, y: 0, z: 0 },
    };
    Plotly.relayout(plotStruct, { 'scene.camera': cam });
    Plotly.relayout(plotEmbed, { 'scene.camera': cam });
  }, 50);
}

function stopOrbit() {
  if (orbitTimer) {
    clearInterval(orbitTimer);
    orbitTimer = null;
  }
}

function applyPresentationMode(on) {
  presenting = on;
  document.body.classList.toggle('presentation', on);
  presentBtn.textContent = on ? 'Exit Presentation' : 'Presentation Mode';

  if (on) {
    viewMode.value = 'both';
    pointSize.value = 10;
    edgeOpacity.value = 0.04;
    showEdges.checked = false;
    showAnn.checked = false;
    startOrbit();
  } else {
    stopOrbit();
  }

  renderPlots();
}

async function loadData() {
  const res = await fetch('data.json');
  const data = await res.json();
  cachedData = data;
  meta.textContent = `method: ${data.meta.label} | nodes: ${data.meta.nodes} | edges: ${data.meta.edges} | learning signal: hop distance`;
  renderPlots();
}

viewMode.addEventListener('change', () => {
  const v = viewMode.value;
  panelStruct.style.display = (v === 'embed') ? 'none' : 'block';
  panelEmbed.style.display = (v === 'struct') ? 'none' : 'block';
  setTimeout(() => {
    renderPlots();
  }, 50);
});

pointSize.addEventListener('input', renderPlots);
edgeOpacity.addEventListener('input', renderPlots);
showEdges.addEventListener('change', renderPlots);
showAnn.addEventListener('change', renderPlots);

resetBtn.addEventListener('click', () => {
  cameraStruct = null;
  cameraEmbed = null;
  renderPlots();
});

presentBtn.addEventListener('click', () => {
  applyPresentationMode(!presenting);
});

loadData();
