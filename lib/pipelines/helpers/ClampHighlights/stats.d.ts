/// <reference types="@webgpu/types" />
import { FrameGate } from "../FrameGate/index.js";
import { Anime4KPipeline } from "../../interfaces.js";
//#region src/pipelines/helpers/ClampHighlights/stats.d.ts
export interface ClampStatsPipelineDescriptor {
  device: GPUDevice;
  /** Исходный кадр. */
  inputTexture: GPUTexture;
  /** Ворота повторов: проход запускается косвенно и на повторе пропускается. */
  gate?: FrameGate;
  name?: string;
}
/**
 * Первая половина Clamp Highlights: максимум яркости 5×5 по исходнику
 * (r32float). Вторую — зажим — выполняет последнее звено цепочки, если ему
 * передан `deRing: stats.getOutputTexture()` (CNN-модели, Downscale): так
 * не нужен отдельный проход в разрешении выхода.
 */
export declare class ClampStats implements Anime4KPipeline {
  name: string;
  pipeline?: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  outputTexture: GPUTexture;
  ready: Promise<void>;
  private launch;
  constructor({ device, inputTexture, gate, name }: ClampStatsPipelineDescriptor);
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}
/**
 * Эпилог зажима для последнего звена: WGSL-функция deRing() и группа
 * привязок 1. Без статистики группа 1 привязана к заглушке, а DERING = false.
 */
export declare class DeRingEpilogue {
  static readonly wgsl: string;
  readonly layout: GPUBindGroupLayout;
  readonly bindGroup: GPUBindGroup;
  readonly constants: Record<string, number>;
  constructor(device: GPUDevice, stats?: GPUTexture);
}
//#endregion