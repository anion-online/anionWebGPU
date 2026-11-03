/// <reference types="@webgpu/types" />
import { FrameGate } from "../helpers/FrameGate/index.js";
import { Anime4KPipeline } from "../interfaces.js";
import { CompactKernel, CompactLayerMeta, CompactModelMeta } from "./shaders.js";
//#region src/pipelines/compact/index.d.ts
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
/**
 * SRVGGNetCompact ×2 (AnimeJaNai V2 SuperUltraCompact, режим «Детали»):
 * слой — compute-проход, признаки между слоями — планарные буферы f16
 * (6 × vec4<f16> на пиксель), выход — текстура rgba16float ×2. С
 * `chromaSigma` перед слоями — два прохода сглаживания цветности кадра, а
 * последний слой берёт от сети только яркость. `SRVGGNetCompactU` — слои
 * на сетке вдвое реже кадра (вход pixel-unshuffle 2×2, выход pixelshuffle ×4).
 */
export declare class CompactSR implements Anime4KPipeline {
  name: string;
  ready: Promise<void>;
  outputTexture: GPUTexture;
  private steps;
  private launch;
  constructor({ device, inputTexture, model, kernel, gate, chromaSigma, chromaThreshold, name }: CompactPipelineDescriptor);
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}
//#endregion
export type { CompactKernel, CompactLayerMeta, CompactModelMeta };