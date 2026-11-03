/**
 * WGSL для SRVGGNetCompact («Детали» в anion-dl, AnimeJaNai V2
 * SuperUltraCompact): 10 свёрток 3×3 с PReLU в разрешении источника,
 * увеличение — pixelshuffle последнего слоя.
 *
 * Ядра (`CompactKernel`):
 * - `reference` — перенос ядра anion-dl (`src/app/player/compact/shaders.ts`,
 *   сверено там с onnxruntime на 72.6 дБ): по пикселю на поток, 24
 *   накопителя f32, `dot` на vec4<f16>, веса из storage-буфера. Эталон для
 *   сверки быстрых ядер.
 * - `fast` — слой как умножение mat4x4<f16> на vec4<f16>: накопители — vec4
 *   по четыре выходных канала, 16 FMA на 16 умножений, без переводов
 *   f16 → f32 на каждом слагаемом; поток — 2×2 пикселя × 3 выходные группы
 *   из 6 (вес читается раз на четыре пикселя), тайл `[группа][пиксель]`; веса слоя — отдельный storage-буфер `array<mat4x4<f16>>`: матрица
 *   читается одной выборкой с одной проверкой границ. Сумма целиком в f16.
 * - `fast-f32acc` — то же, но сумма по каждому отводу ядра (24 слагаемых)
 *   копится в f16 и переносится в накопитель f32: точнее, но медленнее.
 *
 * Замер (Chrome 153, M1 Pro, 1280×720, только сеть): reference 45.6 мс,
 * fast-f32acc 38.1 мс, fast 29.3 мс; раскладка 2×2 × 3 группы (Э14) — ещё
 * −9.5% у fast и −18% у fast-f32acc, выход бит в бит прежний. Проверены и отброшены (docs/PLAN.md, Э10):
 * веса литералами в коде (в 10 раз медленнее), mat4x4<f32>, веса в общей
 * памяти группы, 3–4 пикселя на поток, веса в uniform-буфере (на wgpu −8%, в
 * Chrome +50%: Tint раскладывает mat4x4<f16> в uniform по-своему).
 *
 * Отличия от anion-dl — только ввод-вывод: первый слой читает кадр прямо из
 * текстуры (без прохода препроцесса), последний пишет выход ×2 в текстуру
 * rgba16float, зажатый в 0..1.
 */

/** Каналы пакуются по четыре в vec4<f16>. */
const PACK = 4;

/**
 * `wino` — внутренние слои Winograd F(2×2, 3×3) (`winoLayerShader`), первый и
 * последний — `fast`. Быстрее `fast` только в WebKit (Safari 27: −12%), в
 * Chrome на 23% медленнее — выбор по движку на стороне плеера (Э18).
 */
export type CompactKernel = 'reference' | 'fast' | 'fast-f32acc' | 'wino';

/** Слой из `model.json` конвертера anion-dl; `offset` — в байтах от начала блоба. */
export interface CompactLayerMeta {
  name: string;
  offset: number;
  in: number;
  out: number;
  inGroups: number;
  outGroups: number;
  prelu: boolean;
}

/** `model.json` конвертера anion-dl (`scripts/compact-to-wgsl.py`). */
export interface CompactModelMeta {
  arch: string;
  numFeat: number;
  numConv: number;
  scale: number;
  bytes: number;
  layers: CompactLayerMeta[];
  /**
   * `SRVGGNetCompactU` (Э19): сеть считает на сетке вдвое реже кадра — вход
   * разложен pixel-unshuffle 2×2 (канал c·4 + dy·2 + dx), выход —
   * pixelshuffle ×(2·scale). Конвертер — `training/to_wgsl.py`.
   */
  unshuffle?: number;
  source?: string;
  license?: string;
}

export function groups(channels: number): number {
  return Math.ceil(channels / PACK);
}

interface Geometry {
  /** Потоков по x, y; по z — сколько потоков делят выходные группы пикселя. */
  threads: [number, number, number];
  /** Пикселей на поток по x, y (через шаг в число потоков). */
  block: [number, number];
}

/**
 * Быстрые ядра: поток считает 2×2 пикселя × 3 выходные группы из 6 — весов
 * на умножение читается вдвое меньше, чем при 2 пикселях × 6 группах, а
 * накопителей столько же. Хвост (3 группы) на z не делится — у него 2×1
 * пикселя × все группы; область та же, 32×8.
 */
const GEOMETRY: Record<CompactKernel, Geometry> = {
  reference: { threads: [8, 8, 1], block: [1, 1] },
  fast: { threads: [16, 4, 2], block: [2, 2] },
  'fast-f32acc': { threads: [16, 4, 2], block: [2, 2] },
  // Первый и последний слои `wino` — те же, что у `fast`.
  wino: { threads: [16, 4, 2], block: [2, 2] },
};

const FAST_TAIL: Geometry = { threads: [16, 8, 1], block: [2, 1] };

function geometry(kernel: CompactKernel, tail: boolean): Geometry {
  return tail && kernel !== 'reference' ? FAST_TAIL : GEOMETRY[kernel];
}

/** Сколько пикселей источника покрывает рабочая группа — для диспетча. */
export function kernelRegion(kernel: CompactKernel): [number, number] {
  const { threads, block } = GEOMETRY[kernel];
  return [threads[0] * block[0], threads[1] * block[1]];
}

export interface BlobBases {
  /** Начало весов слоя, в vec4<f16>. */
  w: number;
  bias: number;
  prelu: number;
}

/**
 * Смещения блоков слоя в блобе конвертера, в vec4<f16>: веса
 * `[out][tap][inGroup]`, затем смещения (outGroups vec4), затем наклоны PReLU.
 */
export function layerBases(layer: CompactLayerMeta): BlobBases {
  const w = layer.offset / 8;
  const bias = w + layer.out * 9 * layer.inGroups;
  return { w, bias, prelu: bias + layer.outGroups };
}

/** Число mat4x4 весов слоя у быстрых ядер. */
export function layerMatrices(layer: CompactLayerMeta): number {
  return 9 * layer.inGroups * groups(layer.out);
}

/**
 * Блоб быстрых ядер: по слою — веса `[tap][inGroup][outGroup]` как mat4x4
 * (столбец c — входной канал inGroup·4+c, строка r — выходной outGroup·4+r),
 * затем смещения и наклоны PReLU по outGroups vec4. `order` — индексы f16 в
 * исходном блобе (0xffffffff — ноль: хвост выходных каналов); смещения — в vec4.
 */
export function packMatrices(meta: CompactModelMeta): { order: Uint32Array; bases: BlobBases[] } {
  const order: number[] = [];
  const bases = meta.layers.map((layer) => {
    const src = layerBases(layer);
    const outGroups = groups(layer.out);
    const w = order.length / PACK;
    for (let tap = 0; tap < 9; tap += 1) {
      for (let g = 0; g < layer.inGroups; g += 1) {
        for (let og = 0; og < outGroups; og += 1) {
          for (let c = 0; c < PACK; c += 1) {
            for (let r = 0; r < PACK; r += 1) {
              const o = og * PACK + r;
              order.push(o < layer.out ? (src.w + (o * 9 + tap) * layer.inGroups + g) * PACK + c : -1);
            }
          }
        }
      }
    }
    const bias = order.length / PACK;
    for (let i = 0; i < outGroups * PACK; i += 1) order.push(src.bias * PACK + i);
    const prelu = order.length / PACK;
    if (layer.prelu) for (let i = 0; i < outGroups * PACK; i += 1) order.push(src.prelu * PACK + i);
    return { w, bias, prelu };
  });
  return { order: Uint32Array.from(order, (i) => (i < 0 ? 0xffffffff : i)), bases };
}

export interface LayerShaderOptions {
  kernel: CompactKernel;
  layer: CompactLayerMeta;
  /** Первый слой: вход — кадр rgba8unorm, а не буфер признаков. */
  first: boolean;
  /** Последний слой: pixelshuffle + skip ближайшим соседом в текстуру. */
  tailScale?: number;
  /** Последний слой: цветность — из сглаженной цветности кадра, не от сети. */
  chroma?: boolean;
  /** Сеть на половинной сетке (`SRVGGNetCompactU`), только быстрые ядра. */
  unshuffle?: boolean;
}

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

/** Индекс в тайле: ячейка `pixel` (HW×HH), группа каналов `g`. */
function tileAt(pixel: string, g: string, planar: boolean): string {
  return planar ? `${g} * HW * HH + ${pixel}` : `(${pixel}) * IN_G + ${g}`;
}

/**
 * Тайл активаций с каймой 1 px: кайма за краем кадра — нули, как padding=1 в
 * Conv2d. Обнулять общую память заранее не нужно: тайл пишется целиком.
 * Раскладка: `[пиксель][группа]` или, с `planar`, `[группа][пиксель]` —
 * соседние потоки читают соседние ячейки (`tileAt`).
 */
function loadTile(first: boolean, threads: number, region: [number, number], planar: boolean): string {
  const fill = first
    ? `    var v = V(0.0);
    if (inside) {
      let c = textureLoad(src, vec2i(gx, gy), 0);
      v = V(f16(c.r), f16(c.g), f16(c.b), 0.0h);
    }
    tile[i] = v;`
    : `    if (inside) {
      let s = (u32(gy) * dims.width + u32(gx)) * IN_G;
      for (var g = 0u; g < IN_G; g++) { tile[${tileAt('i', 'g', planar)}] = src[s + g]; }
    } else {
      for (var g = 0u; g < IN_G; g++) { tile[${tileAt('i', 'g', planar)}] = V(0.0); }
    }`;
  return `  let baseX = i32(wg.x * ${region[0]}u) - 1;
  let baseY = i32(wg.y * ${region[1]}u) - 1;
  for (var i = li; i < HW * HH; i += ${threads}u) {
    let gx = baseX + i32(i % HW);
    let gy = baseY + i32(i / HW);
    let inside = gx >= 0 && gy >= 0 && gx < i32(dims.width) && gy < i32(dims.height);
${fill}
  }
  workgroupBarrier();`;
}

/**
 * pixelshuffle (CRD, как в PyTorch: канал = цвет·s² + строка·s + столбец) и
 * skip ближайшим соседом — в srvgg_arch.py стоит `mode='nearest'`. Сумма — в
 * f16, как в anion-dl; выход зажат в 0..1 (там зажимал вывод в rgba8unorm).
 * `at(o)` — выражение f16 для выходного канала o.
 *
 * С `chroma` от сети берётся только яркость, цветность — из сглаженной
 * цветности кадра (`chromaShader`), билинейно: см. CompactPipelineDescriptor.
 */
function storeTail(scale: number, at: (o: number) => string, indent: string, chroma: boolean): string {
  const lines = [
    'let c = textureLoad(orig, vec2i(i32(ox), i32(oy)), 0);',
    'let base = V(f16(c.r), f16(c.g), f16(c.b), 0.0h);',
  ];
  for (let rh = 0; rh < scale; rh += 1) {
    for (let rw = 0; rw < scale; rw += 1) {
      const ch = (colour: number) => at(colour * scale * scale + rh * scale + rw);
      const at2 = `vec2u(ox * ${scale}u + ${rw}u, oy * ${scale}u + ${rh}u)`;
      const rgb = `vec3f(vec3h(${ch(0)} + base.x, ${ch(1)} + base.y, ${ch(2)} + base.z))`;
      if (chroma) {
        lines.push(`{ let p = ${at2};`
          + ` let uv = (vec2f(p) + 0.5) / vec2f(f32(dims.width * ${scale}u), f32(dims.height * ${scale}u));`
          + ` textureStore(dst, p, vec4f(withChroma(${rgb}, textureSampleLevel(chroma, chromaSampler, uv, 0.0).xy), 1.0)); }`);
      } else {
        lines.push(`textureStore(dst, ${at2}, vec4f(saturate(${rgb}), 1.0));`);
      }
    }
  }
  return lines.map((line) => indent + line).join('\n');
}

/** BT.709: так браузер переводит в RGB видео без меток цвета, а Kodik их не ставит. */
const LUMA = { r: 0.2126, b: 0.0722 };

const CHROMA_WGSL = `const KR = ${LUMA.r};
const KB = ${LUMA.b};
const KG = ${1 - LUMA.r - LUMA.b};

fn toChroma(rgb: vec3f) -> vec2f {
  let y = dot(rgb, vec3f(KR, KG, KB));
  return vec2f((rgb.b - y) / (2.0 * (1.0 - KB)), (rgb.r - y) / (2.0 * (1.0 - KR)));
}`;

/** Яркость — из `rgb`, цветность (Cb, Cr) — `cbcr`; результат зажат в 0..1. */
const WITH_CHROMA_WGSL = `${CHROMA_WGSL}

fn withChroma(rgb: vec3f, cbcr: vec2f) -> vec3f {
  let y = dot(rgb, vec3f(KR, KG, KB));
  let r = y + 2.0 * (1.0 - KR) * cbcr.y;
  let b = y + 2.0 * (1.0 - KB) * cbcr.x;
  return saturate(vec3f(r, (y - KR * r - KB * b) / KG, b));
}`;

/** Веса гаусса σ на отводы −R…R, R = ⌈3σ⌉, нормированы на сумму. */
export function gaussTaps(sigma: number): number[] {
  const radius = Math.ceil(3 * sigma);
  const raw = Array.from({ length: 2 * radius + 1 }, (_, i) => Math.exp(-((i - radius) ** 2) / (2 * sigma * sigma)));
  const sum = raw.reduce((a, b) => a + b, 0);
  return raw.map((w) => w / sum);
}

/** Группа цветовых проходов: 8×8 потоков, пиксель на поток. */
export const CHROMA_GROUP = 8;

/**
 * Проход сглаживания цветности в разрешении кадра, по одной оси. Ось `x`
 * читает кадр и переводит его в Cb/Cr, ось `y` — результат оси `x`.
 * Края — повтором крайнего пикселя. Выход rgba16float (rg16float нет среди
 * базовых форматов storage-текстур), заняты xy.
 */
export function chromaShader(axis: 'x' | 'y', sigma: number, threshold = 0): string {
  const taps = gaussTaps(sigma);
  const radius = (taps.length - 1) / 2;
  const at = (q: string) => (axis === 'x' ? `toChroma(textureLoad(src, ${q}, 0).rgb)` : `textureLoad(src, ${q}, 0).xy`);
  const step = axis === 'x' ? 'vec2i(k, 0)' : 'vec2i(0, k)';
  // С порогом — двусторонний вес (Э27): отсчёт, чья цветность дальше от
  // своей на порядок `threshold`, почти не берётся. Блоки сжатия — мелкие
  // перепады, сглаживаются как раньше; цветная линия и край заливки — нет.
  const body = threshold > 0
    ? `  let c = ${at('p')};
  var sum = vec2f(0.0);
  var norm = 0.0;
  for (var i = 0; i < ${taps.length}; i++) {
    let k = i - ${radius};
    let q = clamp(p + ${step}, vec2i(0), size - 1);
    let v = ${at('q')};
    let d = v - c;
    let w = TAPS[i] * exp(-dot(d, d) * ${(1 / (2 * threshold * threshold)).toFixed(4)});
    sum += w * v;
    norm += w;
  }
  textureStore(dst, p, vec4f(sum / norm, 0.0, 1.0));`
    : `  var sum = vec2f(0.0);
  for (var i = 0; i < ${taps.length}; i++) {
    let k = i - ${radius};
    let q = clamp(p + ${step}, vec2i(0), size - 1);
    sum += TAPS[i] * ${at('q')};
  }
  textureStore(dst, p, vec4f(sum, 0.0, 1.0));`;
  return `${CHROMA_WGSL}

const TAPS = array<f32, ${taps.length}>(${taps.map((w) => w.toFixed(8)).join(', ')});

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${CHROMA_GROUP}, ${CHROMA_GROUP}, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(src));
  let p = vec2i(id.xy);
  if (p.x >= size.x || p.y >= size.y) { return; }
${body}
}
`;
}

/** Ядро `reference`: как в anion-dl. */
function referenceMain(layer: CompactLayerMeta, inGroups: number, tailScale: number | undefined, chroma: boolean): string {
  const out = layer.out;
  const lines = [
    '  let ox = wg.x * 8u + lid.x;',
    '  let oy = wg.y * 8u + lid.y;',
    '  // Выход за кадр — только после барьера: соседям нужна загрузка тайла.',
    '  if (ox >= dims.width || oy >= dims.height) { return; }',
    ...range(out).map((o) => `  var a${o} = 0.0f;`),
    `  for (var tap = 0u; tap < 9u; tap++) {
    let p = ((lid.y + tap / 3u) * HW + (lid.x + tap % 3u)) * IN_G;`,
    ...range(out).map((o) => `    let wo${o} = dims.wBase + (${o}u * 9u + tap) * ${inGroups}u;`),
    `    for (var g = 0u; g < IN_G; g++) {
      let x = tile[p + g];`,
    ...range(out).map((o) => `      a${o} += f32(dot(blob[wo${o} + g], x));`),
    '    }',
    '  }',
    ...range(out).map((o) => `  a${o} += f32(blob[dims.biasBase + ${Math.floor(o / PACK)}u][${o % PACK}]);`),
  ];
  if (layer.prelu) {
    lines.push(...range(out).map((o) => `  a${o} = select(f32(blob[dims.preluBase + ${Math.floor(o / PACK)}u][${o % PACK}]) * a${o}, a${o}, a${o} >= 0.0);`));
  }
  if (tailScale) {
    lines.push(storeTail(tailScale, (o) => `f16(a${o})`, '  ', chroma));
  } else {
    lines.push(`  let outBase = (oy * dims.width + ox) * ${groups(out)}u;`);
    for (let og = 0; og < groups(out); og += 1) {
      const parts = range(PACK).map((k) => (og * PACK + k < out ? `f16(a${og * PACK + k})` : '0.0h'));
      lines.push(`  dst[outBase + ${og}u] = V(${parts.join(', ')});`);
    }
  }
  return lines.join('\n');
}

/**
 * Быстрые ядра: mat4x4<f16> × vec4<f16>, несколько пикселей на поток. Поток
 * считает `outG / threads.z` выходных групп: группа k — `lid.z * mine + k`.
 */
function fastMain(
  layer: CompactLayerMeta, inGroups: number, kernel: CompactKernel, tailScale: number | undefined, chroma: boolean,
): string {
  const { threads, block } = geometry(kernel, Boolean(tailScale));
  const [rw, rh] = kernelRegion(kernel);
  const outG = groups(layer.out);
  const split = threads[2];
  if (outG % split) throw new Error(`Compact: ${outG} выходных групп не делятся на ${split} потока`);
  const mine = outG / split;
  // Номер выходной группы k-го накопителя потока.
  const og = (k: number) => (split === 1 ? `${k}u` : `lid.z * ${mine}u + ${k}u`);
  const wide = kernel === 'fast-f32acc';
  const acc = wide ? 'vec4f' : 'V';
  // Пиксели потока — через шаг в число потоков: соседние потоки читают
  // соседние ячейки тайла.
  const px = range(block[0] * block[1]).map((j) => ({ dx: (j % block[0]) * threads[0], dy: Math.floor(j / block[0]) * threads[1] }));
  const sum = wide ? 't' : 'a';

  const lines = [
    ...range(mine).flatMap((k) => px.map((_, j) => `  var a${k}_${j} = ${acc}(0.0);`)),
    '  for (var tap = 0u; tap < 9u; tap++) {',
    '    let ty = lid.y + tap / 3u;',
    '    let tx = lid.x + tap % 3u;',
    ...(wide ? range(mine).flatMap((k) => px.map((_, j) => `    var t${k}_${j} = V(0.0);`)) : []),
    '    for (var g = 0u; g < IN_G; g++) {',
    ...px.map(({ dx, dy }, j) => `      let x${j} = tile[${tileAt(`(ty + ${dy}u) * HW + tx + ${dx}u`, 'g', true)}];`),
    `      let w = (tap * IN_G + g) * ${outG}u;`,
    ...range(mine).flatMap((k) => [
      `      { let m = weights[w + ${og(k)}];`,
      ...px.map((_, j) => `        ${sum}${k}_${j} += m * x${j};`),
      '      }',
    ]),
    '    }',
    ...(wide ? range(mine).flatMap((k) => px.map((_, j) => `    a${k}_${j} += vec4f(t${k}_${j});`)) : []),
    '  }',
  ];

  px.forEach(({ dx, dy }, j) => {
    lines.push(`  {
    let ox = wg.x * ${rw}u + lid.x + ${dx}u;
    let oy = wg.y * ${rh}u + lid.y + ${dy}u;
    if (ox < dims.width && oy < dims.height) {`);
    for (let k = 0; k < mine; k += 1) {
      lines.push(`      var r${k} = a${k}_${j} + ${acc}(blob[dims.biasBase + ${og(k)}]);`);
      if (layer.prelu) {
        lines.push(`      r${k} = select(${acc}(blob[dims.preluBase + ${og(k)}]) * r${k}, r${k}, r${k} >= ${acc}(0.0));`);
      }
    }
    if (tailScale) {
      lines.push(storeTail(tailScale, (o) => `f16(r${Math.floor(o / PACK)}[${o % PACK}])`, '      ', chroma));
    } else {
      lines.push(`      let ob = (oy * dims.width + ox) * ${outG}u;`);
      lines.push(...range(mine).map((k) => `      dst[ob + ${og(k)}] = V(r${k});`));
    }
    lines.push('    }\n  }');
  });
  return lines.join('\n');
}

/**
 * Шейдер слоя. Привязки: 0 — dims, 1 — вход (текстура или буфер признаков),
 * 2 — выход (буфер или текстура), 3 — блоб (у быстрых ядер — только смещения
 * и наклоны PReLU), 4 — кадр для skip (последний слой), 5 — веса слоя
 * матрицами (быстрые ядра), 6 и 7 — сэмплер и сглаженная цветность
 * (последний слой с `chroma`).
 */
export function layerShader({
  kernel, layer, first, tailScale, chroma = false, unshuffle = false,
}: LayerShaderOptions): string {
  if (unshuffle) return unshuffleLayerShader({ kernel, layer, first, tailScale, chroma });
  const withChroma = Boolean(tailScale) && chroma;
  const inGroups = first ? 1 : layer.inGroups;
  const { threads } = geometry(kernel, Boolean(tailScale));
  const region = kernelRegion(kernel);
  const fast = kernel !== 'reference';
  const hw = region[0] + 2;
  const hh = region[1] + 2;

  return `enable f16;
alias V = vec4<f16>;

const HW: u32 = ${hw}u;
const HH: u32 = ${hh}u;
const IN_G: u32 = ${inGroups}u;

struct Dims { width: u32, height: u32, wBase: u32, biasBase: u32, preluBase: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
${first
    ? '@group(0) @binding(1) var src: texture_2d<f32>;'
    : '@group(0) @binding(1) var<storage, read> src: array<V>;'}
${tailScale
    ? '@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;'
    : '@group(0) @binding(2) var<storage, read_write> dst: array<V>;'}
@group(0) @binding(3) var<storage, read> blob: array<V>;
${tailScale ? '@group(0) @binding(4) var orig: texture_2d<f32>;' : ''}
${fast ? '@group(0) @binding(5) var<storage, read> weights: array<mat4x4<f16>>;' : ''}
${withChroma ? `@group(0) @binding(6) var chromaSampler: sampler;
@group(0) @binding(7) var chroma: texture_2d<f32>;

${WITH_CHROMA_WGSL}
` : ''}
var<workgroup> tile: array<V, ${hw * hh * inGroups}>;

@compute @workgroup_size(${threads.join(', ')})
fn main(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
${loadTile(first, threads[0] * threads[1] * threads[2], region, fast)}

${fast ? fastMain(layer, inGroups, kernel, tailScale, withChroma) : referenceMain(layer, inGroups, tailScale, withChroma)}
}
`;
}

// ——— Половинная сетка (`SRVGGNetCompactU`, Э19) ———
//
// Те же быстрые ядра (2×2 пикселя сетки на поток, mat4x4 × vec4, область
// 32×8), но каналов больше: 48 — 12 групп. Выходные группы делят 4 потока по
// z (по 3 группы на поток, как у SUC), а тайл всех входных групп в лимит
// общей памяти 16 КБ не лезет — он грузится частями по ≤ 6 групп.

const U_THREADS: [number, number, number] = [16, 4, 4];
const U_BLOCK: [number, number] = [2, 2];
/** Групп каналов в части тайла: (32 + 2) × (8 + 2) × 6 × 8 Б = 16 320 Б. */
const U_TILE_GROUPS = 6;

function unshuffleLayerShader({
  kernel, layer, first, tailScale, chroma = false,
}: Omit<LayerShaderOptions, 'unshuffle'>): string {
  if (kernel !== 'fast' && kernel !== 'fast-f32acc') {
    throw new Error(`Compact: половинная сетка есть только у ядер fast и fast-f32acc, не у ${kernel}`);
  }
  const withChroma = Boolean(tailScale) && chroma;
  // Первый слой: 3 цвета × 4 пикселя 2×2 — группа на цвет.
  const inGroups = first ? 3 : layer.inGroups;
  const parts = Math.ceil(inGroups / U_TILE_GROUPS);
  const chunk = inGroups / parts;
  if (!Number.isInteger(chunk)) throw new Error(`Compact: ${inGroups} входных групп не делятся на ${parts} части`);
  const outG = groups(layer.out);
  const split = U_THREADS[2];
  if (outG % split) throw new Error(`Compact: ${outG} выходных групп не делятся на ${split} потока`);
  const mine = outG / split;
  if (tailScale && (mine !== 3 || split !== 2 * tailScale)) {
    throw new Error(`Compact: хвост половинной сетки ×${tailScale} — ${outG} выходных групп, нужно ${6 * tailScale}`);
  }
  // Хвост: группа = цвет·4 + строка shuffle, поток z берёт строку z всех трёх
  // цветов — у него весь выход своих пикселей.
  const og = (k: number) => (tailScale ? `${k * split}u + lid.z` : `lid.z * ${mine}u + ${k}u`);
  const [rw, rh] = [U_THREADS[0] * U_BLOCK[0], U_THREADS[1] * U_BLOCK[1]];
  const hw = rw + 2;
  const hh = rh + 2;
  const wide = kernel === 'fast-f32acc';
  const acc = wide ? 'vec4f' : 'V';
  const sum = wide ? 't' : 'a';
  const px = range(U_BLOCK[0] * U_BLOCK[1]).map((j) => ({ dx: (j % U_BLOCK[0]) * U_THREADS[0], dy: Math.floor(j / U_BLOCK[0]) * U_THREADS[1] }));
  const threads = U_THREADS[0] * U_THREADS[1] * U_THREADS[2];

  const fill = first
    ? `    if (inside) {
      let p = vec2i(gx * 2, gy * 2);
      // При нечётной стороне последний столбец (строка) 2×2 выходит за кадр —
      // повтор края; при чётной min ничего не меняет.
      let last = vec2i(textureDimensions(src)) - 1;
      let t0 = textureLoad(src, p, 0);
      let t1 = textureLoad(src, min(p + vec2i(1, 0), last), 0);
      let t2 = textureLoad(src, min(p + vec2i(0, 1), last), 0);
      let t3 = textureLoad(src, min(p + vec2i(1, 1), last), 0);
      tile[i] = V(f16(t0.r), f16(t1.r), f16(t2.r), f16(t3.r));
      tile[HW * HH + i] = V(f16(t0.g), f16(t1.g), f16(t2.g), f16(t3.g));
      tile[2u * HW * HH + i] = V(f16(t0.b), f16(t1.b), f16(t2.b), f16(t3.b));
    } else {
      for (var g = 0u; g < CHUNK; g++) { tile[g * HW * HH + i] = V(0.0); }
    }`
    : `    if (inside) {
      let s = (u32(gy) * dims.width + u32(gx)) * IN_G + part;
      for (var g = 0u; g < CHUNK; g++) { tile[g * HW * HH + i] = src[s + g]; }
    } else {
      for (var g = 0u; g < CHUNK; g++) { tile[g * HW * HH + i] = V(0.0); }
    }`;

  const lines = [
    `  let baseX = i32(wg.x * ${rw}u) - 1;`,
    `  let baseY = i32(wg.y * ${rh}u) - 1;`,
    ...range(mine).flatMap((k) => px.map((_, j) => `  var a${k}_${j} = ${acc}(0.0);`)),
    '  for (var part = 0u; part < IN_G; part += CHUNK) {',
    `    for (var i = li; i < HW * HH; i += ${threads}u) {
    let gx = baseX + i32(i % HW);
    let gy = baseY + i32(i / HW);
    let inside = gx >= 0 && gy >= 0 && gx < i32(dims.width) && gy < i32(dims.height);
${fill}
    }`,
    '    workgroupBarrier();',
    '    for (var tap = 0u; tap < 9u; tap++) {',
    '      let ty = lid.y + tap / 3u;',
    '      let tx = lid.x + tap % 3u;',
    ...(wide ? range(mine).flatMap((k) => px.map((_, j) => `      var t${k}_${j} = V(0.0);`)) : []),
    '      for (var g = 0u; g < CHUNK; g++) {',
    ...px.map(({ dx, dy }, j) => `        let x${j} = tile[g * HW * HH + (ty + ${dy}u) * HW + tx + ${dx}u];`),
    `        let w = (tap * IN_G + part + g) * ${outG}u;`,
    ...range(mine).flatMap((k) => [
      `        { let m = weights[w + ${og(k)}];`,
      ...px.map((_, j) => `          ${sum}${k}_${j} += m * x${j};`),
      '        }',
    ]),
    '      }',
    ...(wide ? range(mine).flatMap((k) => px.map((_, j) => `      a${k}_${j} += vec4f(t${k}_${j});`)) : []),
    '    }',
    '    // Следующая часть перезапишет тайл — сначала все дочитывают эту.',
    '    workgroupBarrier();',
    '  }',
  ];

  px.forEach(({ dx, dy }, j) => {
    lines.push(`  {
    let ox = wg.x * ${rw}u + lid.x + ${dx}u;
    let oy = wg.y * ${rh}u + lid.y + ${dy}u;
    if (ox < dims.width && oy < dims.height) {`);
    for (let k = 0; k < mine; k += 1) {
      lines.push(`      var r${k} = a${k}_${j} + ${acc}(blob[dims.biasBase + ${og(k)}]);`);
      if (layer.prelu) {
        lines.push(`      r${k} = select(${acc}(blob[dims.preluBase + ${og(k)}]) * r${k}, r${k}, r${k} >= ${acc}(0.0));`);
      }
    }
    if (tailScale) {
      lines.push(storeTailU(tailScale, withChroma));
    } else {
      lines.push(`      let ob = (oy * dims.width + ox) * ${outG}u;`);
      lines.push(...range(mine).map((k) => `      dst[ob + ${og(k)}] = V(r${k});`));
    }
    lines.push('    }\n  }');
  });

  return `enable f16;
alias V = vec4<f16>;

const HW: u32 = ${hw}u;
const HH: u32 = ${hh}u;
const IN_G: u32 = ${inGroups}u;
const CHUNK: u32 = ${chunk}u;

struct Dims { width: u32, height: u32, wBase: u32, biasBase: u32, preluBase: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
${first
    ? '@group(0) @binding(1) var src: texture_2d<f32>;'
    : '@group(0) @binding(1) var<storage, read> src: array<V>;'}
${tailScale
    ? '@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;'
    : '@group(0) @binding(2) var<storage, read_write> dst: array<V>;'}
@group(0) @binding(3) var<storage, read> blob: array<V>;
${tailScale ? '@group(0) @binding(4) var orig: texture_2d<f32>;' : ''}
@group(0) @binding(5) var<storage, read> weights: array<mat4x4<f16>>;
${withChroma ? `@group(0) @binding(6) var chromaSampler: sampler;
@group(0) @binding(7) var chroma: texture_2d<f32>;

${WITH_CHROMA_WGSL}
` : ''}
var<workgroup> tile: array<V, ${hw * hh * chunk}>;

@compute @workgroup_size(${U_THREADS.join(', ')})
fn main(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
${lines.join('\n')}
}
`;
}

/**
 * Хвост половинной сетки: pixelshuffle ×s (s = 2·scale, канал =
 * цвет·s² + строка·s + столбец) — поток пишет строку `lid.z` блока s×s
 * пикселя сетки, `r0..r2` — R, G, B этой строки по столбцам. Skip — кадр
 * ближайшим соседом ×scale, как у SUC.
 */
function storeTailU(scale: number, chroma: boolean): string {
  const s = 2 * scale;
  const lines: string[] = [];
  for (let col = 0; col < s; col += 1) {
    const p = `vec2u(ox * ${s}u + ${col}u, oy * ${s}u + lid.z)`;
    const rgb = `vec3f(vec3h(f16(r0[${col}]) + f16(c.r), f16(r1[${col}]) + f16(c.g), f16(r2[${col}]) + f16(c.b)))`;
    const store = chroma
      ? 'let uv = (vec2f(p) + 0.5) / vec2f(textureDimensions(dst));'
        + ` textureStore(dst, p, vec4f(withChroma(${rgb}, textureSampleLevel(chroma, chromaSampler, uv, 0.0).xy), 1.0));`
      : `textureStore(dst, p, vec4f(saturate(${rgb}), 1.0));`;
    // За кадром (нечётная сторона) не пишем: textureStore там и так ничего не
    // делает, но textureLoad из orig вне кадра — неопределённое значение.
    lines.push(`      { let p = ${p}; if (all(p < textureDimensions(dst))) { let c = textureLoad(orig, vec2i(p / ${scale}u), 0); ${store} } }`);
  }
  return lines.join('\n');
}

// ——— Winograd F(2×2, 3×3) для внутренних слоёв (ядро `wino`, Э16, Э18) ———
//
// y = Aᵀ[(G g Gᵀ) ⊙ (Bᵀ d B)]A — корреляция, как Conv2d. Поток — плитка 2×2
// выхода (вход 4×4 из тайла) и 2 выходные группы из 6; 16 позиций ξν — 16
// умножений mat4x4 × vec4 на группу вместо 36 (9 отводов × 4 пикселя).
// Раскладка — вариант лаборатории `wino-z3s`: плиток 8×4, группы делят 3
// потока по z. Выход не бит в бит как у `fast` (другая арифметика), PSNR
// против эталона выше: 67.6 дБ против 65.2.

/** Плиток 2×2 на группу по x, y и потоков по z, делящих выходные группы. */
const WINO_TILES: [number, number] = [8, 4];
const WINO_Z = 3;

/** Сколько пикселей покрывает группа внутреннего слоя `wino` — для диспетча. */
export const WINO_REGION: [number, number] = [2 * WINO_TILES[0], 2 * WINO_TILES[1]];

const G_ROWS = [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]];

function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

function roundHalfEven(x: number): number {
  const floor = Math.floor(x);
  const rest = x - floor;
  if (rest !== 0.5) return rest < 0.5 ? floor : floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Число → f16 с округлением к ближайшему чётному, как Float16Array. */
function floatToHalf(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const a = Math.abs(value);
  if (a >= 65520) return sign | 0x7c00;
  // Субнормали: шаг 2^-24; перенос в 1024 — это ровно наименьшее нормальное.
  if (a < 2 ** -14) return sign | roundHalfEven(a * 2 ** 24);
  let exp = Math.floor(Math.log2(a));
  if (2 ** exp > a) exp -= 1;
  if (2 ** (exp + 1) <= a) exp += 1;
  // Перенос мантиссы в 1024 сам поднимает порядок на единицу.
  return sign | (((exp + 15) << 10) + roundHalfEven((a / 2 ** exp - 1) * 1024));
}

/**
 * Веса внутреннего слоя после G g Gᵀ: mat4x4 `[позиция 16][inGroup][outGroup]`
 * (столбец — входной канал, строка — выходной, как у packMatrices), f16.
 * `raw` — блоб конвертера целиком.
 */
export function winoMatrices(layer: CompactLayerMeta, raw: Uint16Array): Uint16Array {
  const src = layerBases(layer);
  const outGroups = groups(layer.out);
  const w = (o: number, tap: number, i: number): number => (o < layer.out
    ? halfToFloat(raw[(src.w + (o * 9 + tap) * layer.inGroups + Math.floor(i / PACK)) * PACK + (i % PACK)])
    : 0);
  const out: number[] = [];
  for (let xi = 0; xi < 4; xi += 1) {
    for (let nu = 0; nu < 4; nu += 1) {
      for (let g = 0; g < layer.inGroups; g += 1) {
        for (let og = 0; og < outGroups; og += 1) {
          for (let c = 0; c < PACK; c += 1) {
            for (let r = 0; r < PACK; r += 1) {
              let u = 0;
              for (let ky = 0; ky < 3; ky += 1) {
                for (let kx = 0; kx < 3; kx += 1) {
                  u += G_ROWS[xi][ky] * w(og * PACK + r, ky * 3 + kx, g * PACK + c) * G_ROWS[nu][kx];
                }
              }
              out.push(floatToHalf(u));
            }
          }
        }
      }
    }
  }
  return Uint16Array.from(out);
}

/**
 * Внутренний слой (вход и выход — буферы признаков, PReLU) на Winograd.
 * Привязки как у `fast`: 0 — размеры и смещения, 1/2 — признаки, 3 — блоб
 * (смещения и наклоны PReLU), 5 — веса `winoMatrices`.
 */
export function winoLayerShader(layer: CompactLayerMeta): string {
  const inG = layer.inGroups;
  const outG = groups(layer.out);
  const mine = outG / WINO_Z;
  if (!Number.isInteger(mine)) throw new Error(`wino: ${outG} выходных групп не делятся на ${WINO_Z}`);
  const [tx, ty] = WINO_TILES;
  const [rw, rh] = WINO_REGION;
  const hw = rw + 2;
  const hh = rh + 2;
  const threads = tx * ty * WINO_Z;
  const og = (k: number) => `lid.z * ${mine}u + ${k}u`;
  const P = range(16);
  const K = range(mine);
  const lines: string[] = [];
  lines.push(...P.flatMap((p) => K.map((k) => `  var m${p}_${k} = V(0.0);`)));
  lines.push('  let t0 = (2u * lid.y) * HW + 2u * lid.x;');
  lines.push('  for (var g = 0u; g < IN_G; g++) {');
  lines.push(`    let w = g * ${outG}u;`);
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) lines.push(`    let d${r}${c} = tile[g * HW * HH + t0 + ${r * hw + c}u];`);
  }
  // Bᵀd по столбцам: s0 = d0 − d2, s1 = d1 + d2, s2 = d2 − d1, s3 = d1 − d3.
  for (let c = 0; c < 4; c += 1) {
    lines.push(`    let s0${c} = d0${c} - d2${c}; let s1${c} = d1${c} + d2${c}; let s2${c} = d2${c} - d1${c}; let s3${c} = d1${c} - d3${c};`);
  }
  // … затем B по строкам.
  for (let r = 0; r < 4; r += 1) {
    lines.push(`    let v${r * 4} = s${r}0 - s${r}2; let v${r * 4 + 1} = s${r}1 + s${r}2;`);
    lines.push(`    let v${r * 4 + 2} = s${r}2 - s${r}1; let v${r * 4 + 3} = s${r}1 - s${r}3;`);
  }
  for (const p of P) {
    for (const k of K) {
      lines.push(`    m${p}_${k} += weights[${p * inG * outG}u + w + ${og(k)}] * v${p};`);
    }
  }
  lines.push('  }');
  // Aᵀ M A и запись четырёх пикселей плитки.
  lines.push(`  let oy = wg.y * ${rh}u + 2u * lid.y;`);
  lines.push(`  let ox = wg.x * ${rw}u + 2u * lid.x;`);
  for (const k of K) {
    const m = (r: number, c: number) => `m${r * 4 + c}_${k}`;
    lines.push('  {');
    for (let c = 0; c < 4; c += 1) {
      lines.push(`    let a0${c} = ${m(0, c)} + ${m(1, c)} + ${m(2, c)}; let a1${c} = ${m(1, c)} - ${m(2, c)} - ${m(3, c)};`);
    }
    lines.push(`    let bias = blob[dims.biasBase + ${og(k)}];`);
    if (layer.prelu) lines.push(`    let slope = blob[dims.preluBase + ${og(k)}];`);
    for (let i = 0; i < 2; i += 1) {
      [`a${i}0 + a${i}1 + a${i}2`, `a${i}1 - a${i}2 - a${i}3`].forEach((y, j) => {
        lines.push(`    if (ox + ${j}u < dims.width && oy + ${i}u < dims.height) {
      var r = ${y} + bias;${layer.prelu ? '\n      r = select(slope * r, r, r >= V(0.0));' : ''}
      dst[((oy + ${i}u) * dims.width + ox + ${j}u) * ${outG}u + ${og(k)}] = r;
    }`);
      });
    }
    lines.push('  }');
  }

  return `enable f16;
alias V = vec4<f16>;

const HW: u32 = ${hw}u;
const HH: u32 = ${hh}u;
const IN_G: u32 = ${inG}u;

struct Dims { width: u32, height: u32, wBase: u32, biasBase: u32, preluBase: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> src: array<V>;
@group(0) @binding(2) var<storage, read_write> dst: array<V>;
@group(0) @binding(3) var<storage, read> blob: array<V>;
@group(0) @binding(5) var<storage, read> weights: array<mat4x4<f16>>;

var<workgroup> tile: array<V, ${hw * hh * inG}>;

@compute @workgroup_size(${tx}, ${ty}, ${WINO_Z})
fn main(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let baseX = i32(wg.x * ${rw}u) - 1;
  let baseY = i32(wg.y * ${rh}u) - 1;
  for (var i = li; i < HW * HH; i += ${threads}u) {
    let gx = baseX + i32(i % HW);
    let gy = baseY + i32(i / HW);
    let inside = gx >= 0 && gy >= 0 && gx < i32(dims.width) && gy < i32(dims.height);
    if (inside) {
      let s = (u32(gy) * dims.width + u32(gx)) * IN_G;
      for (var g = 0u; g < IN_G; g++) { tile[g * HW * HH + i] = src[s + g]; }
    } else {
      for (var g = 0u; g < IN_G; g++) { tile[g * HW * HH + i] = V(0.0); }
    }
  }
  workgroupBarrier();

${lines.join('\n')}
}
`;
}
