// Стенд для форка anime4k-webgpu.
//
// Две задачи:
//  1. Скорость: GPU-время каждого звена пресета (timestamp-query), медиана и p95.
//  2. Корректность: попиксельное сравнение выхода кандидата с эталоном 1.0.0.
//     Оптимизации, которые не должны менять математику (слияние проходов,
//     тайлинг), обязаны давать PSNR ≈ ∞ / maxDiff ≈ точность f16.

const npmBaseline = window.a4kBaseline;

/**
 * Эталон: 1.0.0 из npm или снимок прошлой сборки форка (bench/ref/<этап>).
 * Снимок нужен для правок, не меняющих математику, после этапов, которые её
 * намеренно меняют (Э1): иначе maxΔ < 1/255 против 1.0.0 недостижим.
 */
async function loadReference() {
  const value = $('reference').value;
  return value === '1.0.0' ? npmBaseline : import(`./${value}/index.js`);
}
const candidateLib = await import('../lib/index.js');

const SCENARIOS = [
  { id: '720-1080', src: [1280, 720], dst: [1920, 1080], checked: true },
  { id: '720-1440', src: [1280, 720], dst: [2560, 1440], checked: true },
  { id: '480-1080', src: [854, 480], dst: [1920, 1080], checked: true },
  // Kodik отдаёт и 853×480: нечётная сторона у половинной сетки (Э24).
  { id: '853-1080', src: [853, 480], dst: [1706, 960], checked: false },
  { id: '1080-1080', src: [1920, 1080], dst: [1920, 1080], checked: false },
  { id: '720-2880', src: [1280, 720], dst: [3840, 2160], checked: false },
];

const $ = (id) => document.getElementById(id);
const log = (line) => { $('log').textContent += `${line}\n`; };

$('scenarios').innerHTML = SCENARIOS.map((s) => `
  <label><input type="checkbox" name="scenario" value="${s.id}" ${s.checked ? 'checked' : ''}>
  ${s.src.join('×')} → ${s.dst.join('×')}</label>`).join('');

// ---------- GPU ----------

/** Ошибки GPU за прогон: при них выход может быть пустым, и сверка врёт. */
let gpuErrors = 0;

async function initDevice() {
  if (!navigator.gpu) throw new Error('WebGPU недоступен');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('Нет адаптера WebGPU');

  const hasTimestamps = adapter.features.has('timestamp-query');
  const hasF16 = adapter.features.has('shader-f16');
  // С флажком shader-f16 кандидат получает precision: 'f16'. Сама фича
  // включается всегда, когда есть: без неё не работают «Детали».
  const useF16 = $('f16').checked && hasF16;
  const device = await adapter.requestDevice({
    requiredFeatures: [
      ...(hasTimestamps ? ['timestamp-query'] : []),
      ...(hasF16 ? ['shader-f16'] : []),
    ],
    // Чтение выхода 4K в rgba32f — ~130 МБ, больше лимита по умолчанию.
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.addEventListener('uncapturederror', (e) => { gpuErrors += 1; log(`GPU error: ${e.error.message}`); });

  const info = adapter.info ?? {};
  $('gpu').textContent = `GPU: ${info.vendor ?? '?'} ${info.architecture ?? ''} ${info.description ?? ''}`
    + ` · timestamp-query: ${hasTimestamps ? 'да' : 'нет (только wall-clock)'}`
    + ` · shader-f16: ${adapter.features.has('shader-f16') ? 'да' : 'нет'}`
    + ` · subgroups: ${adapter.features.has('subgroups') ? 'да' : 'нет'}`;
  return { device, hasTimestamps, useF16 };
}

// ---------- веса «Деталей» ----------

const compactModels = new Map();

/** Эталон «Деталей» — всегда эти веса: с другими сверка показывает разницу моделей. */
const COMPACT_REFERENCE_WEIGHTS = 'janai-v2';

/**
 * Веса SuperUltraCompact из bench/models/<name> (локальные): `janai-v2` —
 * AnimeJaNai V2 как в anion, `kodik-v1` — она же, дообученная под порчу
 * Kodik (Э11, `training/`); `kodik-ink3` / `kodik-ink8` — с весом штриха в
 * обучении (Э13). `u48x7` / `u32x10` — сети половинной сетки с нуля (Э19,
 * `training/to_wgsl.py`).
 */
async function loadCompactModel(name) {
  if (compactModels.has(name)) return compactModels.get(name);
  const file = (f) => `./models/${name}/${f}`;
  const [meta, weights] = await Promise.all([
    fetch(file('model.json')).then((r) => { if (!r.ok) throw new Error(`нет bench/models/${name}/model.json`); return r.json(); }),
    fetch(file('weights.bin')).then((r) => { if (!r.ok) throw new Error(`нет bench/models/${name}/weights.bin`); return r.arrayBuffer(); }),
  ]);
  const model = { meta, weights };
  compactModels.set(name, model);
  return model;
}

/** Опции пресета под режим: веса и ядро «Деталей», модели остальных. */
async function presetOptions(mode, precision) {
  const common = {
    denoiseModel: $('denoiseModel').value, precision, model: $('artModel').value,
  };
  if (mode !== 'ModeCompact') return common;
  return {
    ...common,
    model: await loadCompactModel($('compactWeights').value),
    kernel: $('compactKernel').value,
    chromaSigma: Number($('compactChroma').value),
    chromaThreshold: Number($('compactChromaThreshold').value),
    restore: $('compactRestore').value || undefined,
    deband: $('deband').checked && {
      threshold: Number($('debandThreshold').value),
      range: Number($('debandRange').value),
      iterations: Number($('debandIterations').value),
    },
  };
}

// ---------- источник ----------

/** Синтетический кадр: линии, градиенты, мелкий текст, пережатый в JPEG. */
async function syntheticBitmap() {
  const w = 1920; const h = 1080;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d');
  const sky = ctx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, '#6fa8dc'); sky.addColorStop(1, '#f4cccc');
  ctx.fillStyle = sky; ctx.fillRect(0, 0, w, h);
  ctx.lineCap = 'round';
  for (let i = 0; i < 40; i += 1) {
    ctx.strokeStyle = i % 2 ? '#1b1b1b' : '#3d2b56';
    ctx.lineWidth = 1 + (i % 5);
    ctx.beginPath();
    ctx.moveTo(100 + i * 40, 150);
    ctx.bezierCurveTo(300 + i * 30, 400, 200 + i * 35, 700, 150 + i * 42, 1000);
    ctx.stroke();
  }
  for (let i = 0; i < 12; i += 1) {
    ctx.fillStyle = `hsl(${i * 30} 70% 60%)`;
    ctx.beginPath(); ctx.arc(1300 + (i % 4) * 140, 200 + Math.floor(i / 4) * 220, 60, 0, Math.PI * 2); ctx.fill();
    ctx.lineWidth = 3; ctx.strokeStyle = '#111'; ctx.stroke();
  }
  ctx.fillStyle = '#111'; ctx.font = '22px sans-serif';
  for (let i = 0; i < 6; i += 1) ctx.fillText('Аниме 4K тест — тонкий текст 0123456789', 1150, 820 + i * 34);
  // Kodik отдаёт пережатое видео, не размытое: имитируем сильным JPEG.
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.35 });
  return createImageBitmap(blob);
}

async function sourceBitmap() {
  const file = $('file').files[0];
  if (!file) return syntheticBitmap();
  if (file.type.startsWith('image/')) return createImageBitmap(file);

  const video = document.createElement('video');
  video.muted = true;
  video.src = URL.createObjectURL(file);
  await new Promise((ok, fail) => {
    video.addEventListener('loadeddata', ok, { once: true });
    video.addEventListener('error', fail, { once: true });
  });
  video.currentTime = Math.min(Number($('seek').value) || 0, video.duration - 0.1);
  await new Promise((ok) => { video.addEventListener('seeked', ok, { once: true }); });
  return createImageBitmap(video);
}

function uploadTexture(device, bitmap, [w, h]) {
  // Как в anion: rgba8unorm, кадр масштабируется в нужное «исходное» разрешение.
  const texture = device.createTexture({
    size: [w, h],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return createImageBitmap(bitmap, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' })
    .then((scaled) => {
      device.queue.copyExternalImageToTexture({ source: scaled }, { texture }, [w, h]);
      return texture;
    });
}

// ---------- замер ----------

/** Имя звена. Классы в бандле минифицированы — ищем класс среди экспортов библиотеки. */
function stageName(lib, stage, index) {
  const exported = Object.keys(lib).find((k) => lib[k] === stage.constructor);
  return exported ?? (stage.name || `звено ${index}`);
}

const MARK_WGSL = /* wgsl */ `@compute @workgroup_size(1) fn main() {}`;
const markPipelines = new WeakMap();

/**
 * Compute-проход с timestampWrites — метка времени между звеньями.
 * Проход не пустой и пишет пару меток (2i, 2i+1): пустой проход с одной меткой
 * начала Chrome на Metal оставляет нулевым. Звенья пресета — публичное поле `pipelines`.
 */
function mark(device, encoder, querySet, index) {
  let pipeline = markPipelines.get(device);
  if (!pipeline) {
    pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: MARK_WGSL }), entryPoint: 'main' },
    });
    markPipelines.set(device, pipeline);
  }
  const pass = encoder.beginComputePass({
    timestampWrites: { querySet, beginningOfPassWriteIndex: 2 * index, endOfPassWriteIndex: 2 * index + 1 },
  });
  pass.setPipeline(pipeline);
  pass.dispatchWorkgroups(1);
  pass.end();
}

async function timeStages(device, lib, preset, iterations) {
  const stages = preset.pipelines;
  const count = 2 * (stages.length + 1);
  const querySet = device.createQuerySet({ type: 'timestamp', count });
  const resolve = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const perStage = stages.map(() => []);
  const total = [];
  let invalid = false;
  for (let i = 0; i < iterations; i += 1) {
    const encoder = device.createCommandEncoder();
    mark(device, encoder, querySet, 0);
    stages.forEach((stage, s) => { stage.pass(encoder); mark(device, encoder, querySet, s + 1); });
    encoder.resolveQuerySet(querySet, 0, count, resolve, 0);
    encoder.copyBufferToBuffer(resolve, 0, read, 0, count * 8);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(read.getMappedRange().slice(0));
    read.unmap();
    // Метки обязаны идти по возрастанию; иначе замер стадий недостоверен.
    for (let s = 1; s < count; s += 1) {
      if (t[s] < t[s - 1] || t[s] === 0n) { invalid = true; break; }
    }
    // Звено s — от конца метки s до начала метки s+1.
    for (let s = 0; s < stages.length; s += 1) perStage[s].push(Number(t[2 * s + 2] - t[2 * s + 1]) / 1e6);
    total.push(Number(t[count - 1] - t[0]) / 1e6);
  }
  querySet.destroy(); resolve.destroy(); read.destroy();
  if (invalid) return null;
  return { total, perStage: stages.map((st, s) => ({ name: stageName(lib, st, s), ms: perStage[s] })) };
}

/** Пропускная способность: N кадров подряд без ожидания, как в плеере. */
async function timeWall(device, stages, iterations) {
  const start = performance.now();
  for (let i = 0; i < iterations; i += 1) {
    const encoder = device.createCommandEncoder();
    stages.forEach((stage) => stage.pass(encoder));
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - start) / iterations;
}

/**
 * Цена звеньев по потоку: замер префиксов цепочки (звенья 0..k), цена звена k —
 * разница соседних префиксов. Работает и там, где timestamp-query между
 * проходами врёт (Apple: тайловый GPU перекрывает проходы).
 */
async function timePrefixes(device, lib, preset, iterations) {
  const stages = preset.pipelines;
  const prefix = [];
  for (let k = 1; k <= stages.length; k += 1) prefix.push(await timeWall(device, stages.slice(0, k), iterations));
  return {
    wall: prefix[prefix.length - 1],
    perStage: stages.map((st, k) => ({ name: stageName(lib, st, k), ms: prefix[k] - (k ? prefix[k - 1] : 0) })),
  };
}

/** Медиана GPU-времени прогона или undefined, если метки недостоверны. */
const gpuMedian = (x) => (x?.gpu ? median(x.gpu.total) : undefined);
const median = (xs) => { const s = xs.toSorted((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const p95 = (xs) => { const s = xs.toSorted((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(s.length * 0.95))]; };

// ---------- чтение выхода и сравнение ----------

const READBACK_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) p: vec3u) {
  let d = textureDimensions(src);
  if (p.x >= d.x || p.y >= d.y) { return; }
  dst[p.y * d.x + p.x] = textureLoad(src, p.xy, 0);
}`;

// Пайплайн на каждое устройство: стенд создаёт новое устройство на прогон.
const readbackPipelines = new WeakMap();

/** Выходные текстуры библиотеки без COPY_SRC — читаем через compute в буфер. */
async function readTexture(device, texture) {
  let readbackPipeline = readbackPipelines.get(device);
  if (!readbackPipeline) {
    readbackPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: READBACK_WGSL }), entryPoint: 'main' },
    });
    readbackPipelines.set(device, readbackPipeline);
  }
  const { width, height } = texture;
  const size = width * height * 16;
  const storage = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({
    layout: readbackPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: texture.createView() }, { binding: 1, resource: { buffer: storage } }],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(readbackPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
  pass.end();
  encoder.copyBufferToBuffer(storage, 0, read, 0, size);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(read.getMappedRange().slice(0));
  read.unmap(); storage.destroy(); read.destroy();
  return { width, height, data };
}

/**
 * Рамка у края кадра, которую можно не учитывать в вердикте. Нужна для сверки
 * с эталонами до Э2: там слои читали за границей текстуры (на Metal — нули), а
 * генератор Э2 прижимает координаты к краю, как texOff в mpv. Разница
 * проникает внутрь на рецептивное поле сети: 7 слоёв 3×3 → 14 px выхода x2.
 */
const EDGE = 16;

function compare(a, b, ignoreEdge) {
  if (a.width !== b.width || a.height !== b.height) {
    return { text: `размер разный: ${a.width}×${a.height} vs ${b.width}×${b.height}`, ok: false };
  }
  let maxDiff = 0; let maxInner = 0; let sq = 0; let changed = 0; const n = a.width * a.height;
  for (let i = 0; i < n; i += 1) {
    const x = i % a.width; const y = Math.floor(i / a.width);
    const inner = x >= EDGE && y >= EDGE && x < a.width - EDGE && y < a.height - EDGE;
    let pixelMax = 0;
    for (let c = 0; c < 3; c += 1) {
      const d = Math.abs(Math.min(1, Math.max(0, a.data[i * 4 + c])) - Math.min(1, Math.max(0, b.data[i * 4 + c])));
      pixelMax = Math.max(pixelMax, d); sq += d * d;
    }
    maxDiff = Math.max(maxDiff, pixelMax);
    if (inner) maxInner = Math.max(maxInner, pixelMax);
    if (pixelMax >= 1 / 255) changed += 1;
  }
  const mse = sq / (n * 3);
  const psnr = mse === 0 ? Infinity : 10 * Math.log10(1 / mse);
  // Шаг f16 около 1.0 — ~0.0005; разница в 1/255 уже видна в 8-битном выводе.
  const ok = (ignoreEdge ? maxInner : maxDiff) < 1 / 255;
  // Доля пикселей, разошедшихся на ≥1/255 — для правок, меняющих картинку
  // намеренно (Э1): разница должна быть точечной, а не по всему кадру.
  const share = (changed / n) * 100;
  return {
    text: `maxΔ ${(maxDiff * 255).toFixed(2)}/255 (без края ${EDGE}px: ${(maxInner * 255).toFixed(2)}) · PSNR ${psnr === Infinity ? '∞' : psnr.toFixed(1)} дБ · Δ≥1/255: ${share.toFixed(2)}% пикс.`,
    ok,
  };
}

function show(title, img, diffWith) {
  const canvas = document.createElement('canvas');
  canvas.width = img.width; canvas.height = img.height;
  const pixels = new ImageData(img.width, img.height);
  for (let i = 0; i < img.width * img.height; i += 1) {
    for (let c = 0; c < 3; c += 1) {
      const v = diffWith
        ? Math.abs(img.data[i * 4 + c] - diffWith.data[i * 4 + c]) * 16 // усиление разницы
        : img.data[i * 4 + c];
      pixels.data[i * 4 + c] = Math.round(Math.min(1, Math.max(0, v)) * 255);
    }
    pixels.data[i * 4 + 3] = 255;
  }
  canvas.getContext('2d').putImageData(pixels, 0, 0);
  const figure = document.createElement('figure');
  figure.innerHTML = `<figcaption class="muted">${title}</figcaption>`;
  figure.append(canvas);
  $('views').append(figure);
}

// ---------- прогон ----------

function buildPreset(lib, mode, device, input, src, dst, extra = {}) {
  return new lib[mode]({
    device,
    inputTexture: input,
    nativeDimensions: { width: src[0], height: src[1] },
    targetDimensions: { width: dst[0], height: dst[1] },
    ...extra,
  });
}

async function measure(device, hasTimestamps, lib, preset, iterations) {
  for (let i = 0; i < 5; i += 1) { // прогрев: компиляция пайплайнов, кэши
    const enc = device.createCommandEncoder(); preset.pass(enc); device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const { wall, perStage } = await timePrefixes(device, lib, preset, iterations);
  const gpu = hasTimestamps ? await timeStages(device, lib, preset, iterations) : null;
  return { wall, perStage, gpu };
}

/**
 * Сборка пресета до первого готового кадра: сколько заняло и самая долгая
 * заморозка основного потока за это время (по пропускам 4-мс таймера). Звенья
 * форка компилируют шейдеры в фоне (ready), 1.0.0 — синхронно.
 */
async function build(lib, mode, device, input, src, dst, extra) {
  let maxGap = 0; let last = performance.now();
  const timer = setInterval(() => {
    const now = performance.now(); maxGap = Math.max(maxGap, now - last); last = now;
  }, 4);
  const start = performance.now();
  const preset = buildPreset(lib, mode, device, input, src, dst, extra);
  await preset.ready;
  const enc = device.createCommandEncoder(); preset.pass(enc); device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  await new Promise((r) => { setTimeout(r, 20); });
  clearInterval(timer);
  return { preset, buildMs: performance.now() - start - 20, maxGap: Math.max(maxGap, performance.now() - last) };
}

function gateOptions() {
  return { skipUnchanged: $('skipUnchanged').checked, unchangedThreshold: Number($('threshold').value) || 0 };
}

function fmt(ms) { return ms === undefined ? '—' : ms.toFixed(2); }

async function run() {
  $('log').textContent = ''; $('results').innerHTML = ''; $('views').innerHTML = '';
  $('run').disabled = true;
  gpuErrors = 0;
  try {
    const { device, hasTimestamps, useF16 } = await initDevice();
    const precision = useF16 ? 'f16' : 'f32';
    const bitmap = await sourceBitmap();
    const iterations = Math.max(5, Number($('iters').value) || 60);
    const modes = [...document.querySelectorAll('input[name=mode]:checked')].map((e) => e.value);
    const scenarios = [...document.querySelectorAll('input[name=scenario]:checked')]
      .map((e) => SCENARIOS.find((s) => s.id === e.value));
    const withBaseline = $('baseline').checked;
    const denoiseModel = $('denoiseModel').value;
    const artModel = $('artModel').value;
    const baselineLib = withBaseline ? await loadReference() : null;
    if (withBaseline) log(`эталон: ${$('reference').value}`);

    const rows = [];
    for (const sc of scenarios) {
      const input = await uploadTexture(device, bitmap, sc.src);
      for (const mode of modes) {
        const compact = mode === 'ModeCompact';
        const weights = $('compactWeights').value;
        const size = mode === 'ModeArtCNN' ? artModel
          : compact ? [
            $('compactKernel').value,
            weights !== COMPACT_REFERENCE_WEIGHTS ? weights : '',
            `σ${$('compactChroma').value}${Number($('compactChromaThreshold').value) > 0 ? `t${$('compactChromaThreshold').value}` : ''}`,
            $('compactRestore').value ? `restore-${$('compactRestore').value}` : '',
            $('deband').checked ? `deband${$('debandThreshold').value}r${$('debandRange').value}i${$('debandIterations').value}` : '',
          ].filter(Boolean).join('/')
          : (denoiseModel !== 'VL' && denoiseModel) || '';
        const variant = [size, useF16 && !compact ? 'f16' : ''].filter(Boolean).join('/');
        const label = `${mode}${variant ? `/${variant}` : ''} ${sc.src.join('×')}→${sc.dst.join('×')}`;
        log(`▶ ${label}`);

        // Эталон всегда VL; при M/L сверка показывает разницу моделей, а не ошибку.
        // С пропуском повторов на неподвижном кадре считается только первый:
        // замер показывает цену повтора (сравнение кадров), а не цепочки.
        const options = await presetOptions(mode, precision);
        const built = await build(candidateLib, mode, device, input, sc.src, sc.dst, {
          ...options, ...gateOptions(),
        });
        const cand = built.preset;
        log(`  сборка кандидата: ${fmt(built.buildMs)} мс, самая долгая заморозка страницы ${fmt(built.maxGap)} мс`);
        const candRes = await measure(device, hasTimestamps, candidateLib, cand, iterations);
        let baseRes = null; let verdict = null;

        // Режима может не быть у эталона (ArtCNN — только в форке). Эталон
        // «Деталей» — ядро reference из той же сборки (перенос anion-dl) с
        // весами JaNai V2: при других весах сверка — это A/B моделей.
        const refLib = compact ? candidateLib : baselineLib;
        const refOptions = compact
          ? { ...options, kernel: 'reference', model: await loadCompactModel(COMPACT_REFERENCE_WEIGHTS) }
          : {};
        if (withBaseline && refLib[mode]) {
          const builtBase = await build(refLib, mode, device, input, sc.src, sc.dst, refOptions);
          const base = builtBase.preset;
          log(`  сборка эталона: ${fmt(builtBase.buildMs)} мс, самая долгая заморозка страницы ${fmt(builtBase.maxGap)} мс`);
          baseRes = await measure(device, hasTimestamps, refLib, base, iterations);
          const [a, b] = await Promise.all([
            readTexture(device, cand.getOutputTexture()),
            readTexture(device, base.getOutputTexture()),
          ]);
          verdict = compare(a, b, $('ignoreEdge').checked);
          if (gpuErrors) verdict = { text: `GPU-ошибки (${gpuErrors}) — сверка недостоверна`, ok: false };
          if (sc === scenarios[0]) {
            show(`${label} · эталон ${compact ? `reference/${COMPACT_REFERENCE_WEIGHTS}` : $('reference').value}`, b);
            show(`${label} · кандидат`, a);
            show(`${label} · |Δ| ×16`, a, b);
          }
        }

        rows.push({ label, cand: candRes, base: baseRes, verdict, out: cand.getOutputTexture() });
        log(`  стадии кандидата (поток): ${candRes.perStage.map((s) => `${s.name} ${fmt(s.ms)}`).join(' · ')}`);
        if (hasTimestamps && !candRes.gpu) log('  timestamp-query: метки недостоверны, GPU-время не показано');
      }
      input.destroy();
    }

    $('results').innerHTML = `
      <table>
        <tr><th>сценарий</th><th>выход</th>
          <th>кандидат GPU, мс (медиана / p95)</th><th>кандидат поток, мс/кадр</th>
          <th>эталон GPU, мс</th><th>эталон поток, мс/кадр</th><th>ускорение (поток)</th><th>сверка</th></tr>
        ${rows.map((r) => {
          // Ускорение — по потоку: он есть всегда и одинаково меряется у обоих.
          return `<tr><td>${r.label}</td><td>${r.out.width}×${r.out.height}</td>
            <td>${r.cand.gpu ? `${fmt(gpuMedian(r.cand))} / ${fmt(p95(r.cand.gpu.total))}` : '—'}</td>
            <td>${fmt(r.cand.wall)}</td>
            <td>${fmt(gpuMedian(r.base))}</td>
            <td>${fmt(r.base?.wall)}</td>
            <td>${r.base ? `${(r.base.wall / r.cand.wall).toFixed(2)}×` : '—'}</td>
            <td class="${r.verdict ? (r.verdict.ok ? 'ok' : 'bad') : ''}">${r.verdict?.text ?? '—'}</td></tr>`;
        }).join('')}
      </table>
      <p class="muted">Бюджет кадра: 41.7 мс при 24 к/с, 33.3 мс при 30 к/с. «Поток» — кадры подряд без ожидания,
      ближе к реальному плееру; стадии в логе — разности префиксов цепочки по потоку.
      GPU-время — по timestamp-query; на Apple оно занижено (проходы перекрываются).</p>`;
    log('готово');
    device.destroy();
  } catch (error) {
    log(`ошибка: ${error?.message ?? error}`);
  } finally {
    $('run').disabled = false;
  }
}

// ---------- прогон ролика ----------

/**
 * Покадровый проход по ролику: перемотка в середину каждого кадра и `onFrame`.
 * Не воспроизведение: requestVideoFrameCallback не срабатывает на скрытой
 * странице, а при воспроизведении кадры, пришедшие пока стенд ждал GPU,
 * терялись бы и занижали долю повторов. Нужна частота кадров ролика.
 */
async function stepFrames(video, start, seconds, fps, onFrame) {
  const count = Math.floor(seconds * fps);
  for (let n = 0; n < count; n += 1) {
    video.currentTime = start + (n + 0.5) / fps;
    await new Promise((ok) => { video.addEventListener('seeked', ok, { once: true }); });
    await onFrame();
  }
}

/** Гистограмма max |Δ| кадра с опорным, в уровнях 8 бит. */
const DIFF_BUCKETS = [[0, 0], [1, 1], [2, 2], [3, 4], [5, 8], [9, 32], [33, Infinity]];

/**
 * Реальный ролик: доля повторов и время кадра (копия + цепочка, до
 * onSubmittedWorkDone — как обратное давление в anion) без ворот и с ними.
 */
async function runVideo() {
  $('log').textContent = ''; $('results').innerHTML = ''; $('views').innerHTML = '';
  $('run').disabled = true; $('runVideo').disabled = true;
  gpuErrors = 0;
  try {
    const file = $('file').files[0];
    if (!file?.type.startsWith('video/')) throw new Error('выберите видеофайл');
    const { device, useF16 } = await initDevice();
    const precision = useF16 ? 'f16' : 'f32';
    const video = document.createElement('video');
    video.muted = true; video.playsInline = true;
    video.src = URL.createObjectURL(file);
    await new Promise((ok, fail) => {
      video.addEventListener('loadeddata', ok, { once: true });
      video.addEventListener('error', fail, { once: true });
    });
    const src = [video.videoWidth, video.videoHeight];
    const sc = SCENARIOS.find((s) => s.id === document.querySelector('input[name=scenario]:checked')?.value) ?? SCENARIOS[0];
    const k = sc.dst[0] / sc.src[0];
    const dst = [Math.round(src[0] * k), Math.round(src[1] * k)];
    const seconds = Math.max(2, Number($('duration').value) || 20);
    const seek = Math.min(Number($('seek').value) || 0, video.duration - seconds);
    const threshold = Number($('threshold').value) || 0;
    const fps = Number($('fps').value) || 23.976;
    const modes = [...document.querySelectorAll('input[name=mode]:checked')].map((e) => e.value);
    const input = device.createTexture({
      size: src,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    log(`ролик ${src.join('×')} → ${dst.join('×')}, ${seconds} с с ${seek.toFixed(1)} с, ${fps} к/с, порог ${threshold}`);

    const rows = [];
    for (const mode of modes) {
      for (const skipUnchanged of [false, true]) {
        const label = `${mode}${useF16 ? '/f16' : ''} · ${skipUnchanged ? `с воротами (порог ${threshold})` : 'без ворот'}`;
        log(`▶ ${label}`);
        const { preset } = await build(candidateLib, mode, device, input, src, dst, {
          ...(await presetOptions(mode, precision)),
          skipUnchanged, unchangedThreshold: threshold,
        });
        // Сборка прогнала один кадр; счётчики ворот — с начала ролика.
        const before = preset.gate ? await preset.gate.readStats() : null;
        const times = []; const hist = DIFF_BUCKETS.map(() => 0);
        await stepFrames(video, seek, seconds, fps, async () => {
          const start = performance.now();
          device.queue.copyExternalImageToTexture({ source: video }, { texture: input }, src);
          const encoder = device.createCommandEncoder();
          preset.pass(encoder);
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
          times.push(performance.now() - start);
          if (preset.gate) {
            const { lastDiff } = await preset.gate.readStats();
            hist[DIFF_BUCKETS.findIndex(([lo, hi]) => lastDiff >= lo && lastDiff <= hi)] += 1;
          }
        });
        let skipped = null;
        if (preset.gate) {
          const after = await preset.gate.readStats();
          skipped = (after.skipped - before.skipped) / (after.frames - before.frames);
          log(`  max |Δ| с опорным (уровни 8 бит): ${DIFF_BUCKETS.map(([lo, hi], i) => `${lo === hi ? lo : `${lo}–${hi === Infinity ? '∞' : hi}`}: ${hist[i]}`).join(' · ')}`);
        }
        const mean = times.reduce((a, b) => a + b, 0) / times.length;
        rows.push({ label, frames: times.length, mean, p95: p95(times), skipped });
      }
    }
    input.destroy();
    if (gpuErrors) log(`GPU-ошибки: ${gpuErrors} — замер недостоверен`);

    $('results').innerHTML = `
      <table>
        <tr><th>прогон</th><th>кадров</th><th>кадр, мс (среднее / p95)</th><th>повторов</th></tr>
        ${rows.map((r) => `<tr><td>${r.label}</td><td>${r.frames}</td>
          <td>${fmt(r.mean)} / ${fmt(r.p95)}</td>
          <td>${r.skipped === null ? '—' : `${(r.skipped * 100).toFixed(1)}%`}</td></tr>`).join('')}
      </table>
      <p class="muted">«Кадр» — копия кадра + цепочка до onSubmittedWorkDone, как в цикле anion.
      Ролик проходится покадрово перемоткой, без потерь кадров; среднее по всем кадрам
      учитывает и пропущенные воротами.</p>`;
    log('готово');
    device.destroy();
  } catch (error) {
    log(`ошибка: ${error?.message ?? error}`);
  } finally {
    $('run').disabled = false; $('runVideo').disabled = false;
  }
}

$('run').addEventListener('click', run);
$('runVideo').addEventListener('click', runVideo);
// Модуль с top-level await загружен.
$('run').disabled = false;
$('runVideo').disabled = false;
