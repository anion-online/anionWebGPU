import { Anime4KPipeline } from '../interfaces';
import { FrameGate, Launch, launcher } from '../helpers/FrameGate';
import {
  CHROMA_GROUP, chromaShader, CompactKernel, CompactModelMeta, groups, kernelRegion, layerBases, layerMatrices,
  layerShader, packMatrices, WINO_REGION, winoLayerShader, winoMatrices,
} from './shaders';

export type { CompactKernel, CompactLayerMeta, CompactModelMeta } from './shaders';

/** Веса и описание модели — как их выдаёт конвертер anion-dl. */
export interface CompactModelData {
  meta: CompactModelMeta;
  /** `weights.bin`: f16 в порядке чтения шейдера. */
  weights: ArrayBuffer;
}

export interface CompactPipelineDescriptor {
  /** Нужна фича 'shader-f16'. */
  device: GPUDevice;
  inputTexture: GPUTexture;
  model: CompactModelData;
  /**
   * Ядро, по умолчанию `fast`. `reference` — как в anion-dl (эталон),
   * `fast-f32acc` — быстрое с суммой в f32, точнее и медленнее. `wino` —
   * внутренние слои на Winograd: быстрее `fast` в WebKit, медленнее в Chrome.
   */
  kernel?: CompactKernel;
  /** Ворота повторов: слои запускаются косвенно и на повторе пропускаются. */
  gate?: FrameGate;
  /**
   * Цветность выхода. 0 (по умолчанию) — как у сети. Больше нуля — от сети
   * берётся только яркость, а цветность — из кадра, сглаженного гауссом с
   * этим σ (в пикселях кадра) и растянутого билинейно. Сеть выдаёт сразу RGB
   * и цветовые блоки сжатия (у Kodik цветность в половинном разрешении)
   * рисует пятнами, как детали; σ = 2 их убирает (docs/PLAN.md, Э12).
   */
  chromaSigma?: number;
  /**
   * Порог сглаживания цветности (доли 0..1 по Cb/Cr), 0 — без порога, как в
   * Э12. Больше нуля — двусторонний вес: перепады цвета крупнее порога не
   * усредняются, и цветная линия на фоне той же яркости не расплывается в
   * ореол. 0.02 — блоки гаснут как при σ2, края втрое резче (Э27).
   */
  chromaThreshold?: number;
  name?: string;
}

interface Step {
  pipeline?: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  /** Своя сетка запуска (цветовые проходы); без неё — сетка слоёв. */
  launch?: Launch;
}

/**
 * SRVGGNetCompact ×2 (AnimeJaNai V2 SuperUltraCompact, режим «Детали»):
 * слой — compute-проход, признаки между слоями — планарные буферы f16
 * (6 × vec4<f16> на пиксель), выход — текстура rgba16float ×2. С
 * `chromaSigma` перед слоями — два прохода сглаживания цветности кадра, а
 * последний слой берёт от сети только яркость. `SRVGGNetCompactU` — слои
 * на сетке вдвое реже кадра (вход pixel-unshuffle 2×2, выход pixelshuffle ×4).
 */
export class CompactSR implements Anime4KPipeline {
  name: string;

  ready: Promise<void>;

  outputTexture: GPUTexture;

  private steps: Step[] = [];

  private launch: Launch;

  constructor({
    device, inputTexture, model, kernel = 'fast', gate, chromaSigma = 0, chromaThreshold = 0, name = 'CompactSR',
  }: CompactPipelineDescriptor) {
    this.name = name;
    const { meta, weights } = model;
    // `SRVGGNetCompactU` (Э19) — та же сеть на сетке вдвое реже кадра.
    const unshuffle = meta.arch === 'SRVGGNetCompactU';
    if (!unshuffle && meta.arch !== 'SRVGGNetCompact') throw new Error(`${name}: архитектура ${meta.arch} не поддерживается`);
    if (unshuffle && meta.unshuffle !== 2) throw new Error(`${name}: unshuffle ${meta.unshuffle} не поддерживается`);
    if (weights.byteLength !== meta.bytes) {
      throw new Error(`${name}: веса ${weights.byteLength} Б, в описании ${meta.bytes} Б`);
    }
    if (!device.features.has('shader-f16')) throw new Error(`${name}: нужна фича 'shader-f16'`);

    const { width, height } = inputTexture;
    const { scale } = meta;
    if (unshuffle && (kernel === 'reference' || kernel === 'wino')) throw new Error(`${name}: у половинной сетки нет ядра ${kernel}`);
    // Сетка, на которой считают слои.
    // Нечётная сторона (854×480 у Kodik бывает и 853×480): лишний столбец
    // сетки читает край кадра повтором, выход за кадром не пишется.
    const gridW = unshuffle ? Math.ceil(width / 2) : width;
    const gridH = unshuffle ? Math.ceil(height / 2) : height;
    // У `wino` первый и последний слои — ядра `fast`, внутренние — Winograd.
    const wino = kernel === 'wino';
    const direct: CompactKernel = wino ? 'fast' : kernel;
    const [regionX, regionY] = kernelRegion(direct);
    // У всех слоёв одна сетка групп — один слот косвенного запуска на все.
    // Внутренним слоям Winograd — своя: их область вдвое уже.
    this.launch = launcher(gate, Math.ceil(gridW / regionX), Math.ceil(gridH / regionY));
    const winoLaunch = wino
      ? launcher(gate, Math.ceil(width / WINO_REGION[0]), Math.ceil(height / WINO_REGION[1]))
      : undefined;

    this.outputTexture = device.createTexture({
      label: `${name}: output`,
      size: [width * scale, height * scale, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const featBytes = gridW * gridH * groups(meta.numFeat) * 8;
    const ping = device.createBuffer({ label: `${name}: ping`, size: featBytes, usage: storage });
    const pong = device.createBuffer({ label: `${name}: pong`, size: featBytes, usage: storage });

    // Быстрым ядрам веса переложены матрицами (packMatrices): f16 берутся
    // битами, без пересчёта. Смещения слоёв — свои у каждой раскладки.
    const fast = kernel !== 'reference';
    const raw = new Uint16Array(weights);
    const packed = fast ? packMatrices(meta) : null;
    const halves = packed
      ? Uint16Array.from(packed.order, (i) => (i === 0xffffffff ? 0 : raw[i]))
      : raw;
    const blob = device.createBuffer({ label: `${name}: weights`, size: halves.byteLength, usage: storage });
    device.queue.writeBuffer(blob, 0, halves);

    const compiling: Promise<void>[] = [];

    // Цветность — до слоёв: последний слой читает её при записи выхода.
    const chroma = chromaSigma > 0;
    let chromaView: GPUTextureView | undefined;
    let chromaSampler: GPUSampler | undefined;
    if (chroma) {
      const chromaTexture = (label: string) => device.createTexture({
        label: `${name}: ${label}`,
        size: [width, height, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      });
      const across = chromaTexture('chroma x');
      const blurred = chromaTexture('chroma');
      chromaView = blurred.createView();
      chromaSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
      const chromaLaunch = launcher(gate, Math.ceil(width / CHROMA_GROUP), Math.ceil(height / CHROMA_GROUP));
      ([['x', inputTexture, across], ['y', across, blurred]] as const).forEach(([axis, from, to]) => {
        const step: Step = { bindGroup: undefined as unknown as GPUBindGroup, launch: chromaLaunch };
        this.steps.push(step);
        compiling.push(device.createComputePipelineAsync({
          label: `${name}: chroma ${axis}`,
          layout: 'auto',
          compute: {
            module: device.createShaderModule({ label: `${name}: chroma ${axis}`, code: chromaShader(axis, chromaSigma, chromaThreshold) }),
            entryPoint: 'main',
          },
        }).then((pipeline) => {
          step.pipeline = pipeline;
          step.bindGroup = device.createBindGroup({
            label: `${name}: chroma ${axis}`,
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: from.createView() },
              { binding: 1, resource: to.createView() },
            ],
          });
        }));
      });
    }
    let src: GPUBuffer | null = null;
    let dst = ping;
    meta.layers.forEach((layer, index) => {
      const first = index === 0;
      const last = index === meta.layers.length - 1;
      const inner = wino && !first && !last;
      const bases = packed ? packed.bases[index] : layerBases(layer);

      const dims = device.createBuffer({
        label: `${name}: dims ${index}`,
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(dims, 0, new Uint32Array([gridW, gridH, bases.w, bases.bias, bases.prelu, 0, 0, 0]));

      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: dims } },
        { binding: 1, resource: first ? inputTexture.createView() : { buffer: src as GPUBuffer } },
        { binding: 2, resource: last ? this.outputTexture.createView() : { buffer: dst } },
      ];
      entries.push({ binding: 3, resource: { buffer: blob } });
      if (last) entries.push({ binding: 4, resource: inputTexture.createView() });
      if (last && chromaView && chromaSampler) {
        entries.push({ binding: 6, resource: chromaSampler }, { binding: 7, resource: chromaView });
      }
      if (inner) {
        // Веса Winograd (G g Gᵀ) считаются здесь из исходного блоба: 16
        // позиций вместо 9 отводов, смещения и PReLU — из общего блоба.
        const transformed = winoMatrices(layer, raw);
        const matrices = device.createBuffer({ label: `${name}: weights ${index}`, size: transformed.byteLength, usage: storage });
        device.queue.writeBuffer(matrices, 0, transformed);
        entries.push({ binding: 5, resource: { buffer: matrices } });
      } else if (packed) {
        // Веса слоя — отдельным буфером матриц: выборка mat4x4 целиком, с
        // одной проверкой границ (из общего блоба по vec4 — четыре).
        const size = layerMatrices(layer) * 32;
        const matrices = device.createBuffer({ label: `${name}: weights ${index}`, size, usage: storage });
        device.queue.writeBuffer(matrices, 0, halves, bases.w * 4, size / 2);
        entries.push({ binding: 5, resource: { buffer: matrices } });
      }

      const module = device.createShaderModule({
        label: `${name}: layer ${index}`,
        code: inner
          ? winoLayerShader(layer)
          : layerShader({
            kernel: direct, layer, first, tailScale: last ? scale : undefined, chroma, unshuffle,
          }),
      });
      // Раскладка 'auto' видит только привязки, которые шейдер читает, —
      // группа собирается после компиляции.
      const step: Step = { bindGroup: undefined as unknown as GPUBindGroup, launch: inner ? winoLaunch : undefined };
      this.steps.push(step);
      compiling.push(device.createComputePipelineAsync({
        label: `${name}: layer ${index}`,
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      }).then((pipeline) => {
        step.pipeline = pipeline;
        step.bindGroup = device.createBindGroup({
          label: `${name}: layer ${index}`,
          layout: pipeline.getBindGroupLayout(0),
          entries,
        });
      }));

      if (!last) {
        src = dst;
        dst = dst === ping ? pong : ping;
      }
    });
    this.ready = Promise.all(compiling).then(() => undefined);
  }

  updateParam(param: string, value: any): void {
    throw new Error(`${this.name} has no param.`);
  }

  pass(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: this.name });
    this.steps.forEach((step) => {
      if (!step.pipeline) throw new Error(`${this.name}: шейдеры ещё компилируются — дождитесь ready.`);
      pass.setPipeline(step.pipeline);
      pass.setBindGroup(0, step.bindGroup);
      (step.launch ?? this.launch).dispatch(pass);
    });
    pass.end();
  }

  getOutputTexture(): GPUTexture {
    return this.outputTexture;
  }
}
