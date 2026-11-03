/// <reference types="@webgpu/types" />
import { FrameGate } from "../../helpers/FrameGate/index.js";
import { Anime4KPipeline } from "../../interfaces.js";
//#region src/pipelines/denoise/Deband/index.d.ts
export interface DebandParams {
  /**
   * Порог в уровнях 8 бит: пиксель заменяется средним отсчётов, только если
   * отличается от него меньше. Ступень градиента после x264 — 1–2 уровня,
   * линия — десятки: края остаются, полосы сглаживаются.
   */
  threshold?: number;
  /** Радиус выборки первой итерации, пикселей входа. */
  range?: number;
  /** Итераций: каждая — радиус × i, порог / i (как в mpv). */
  iterations?: number;
}
export interface DebandPipelineDescriptor extends DebandParams {
  device: GPUDevice;
  inputTexture: GPUTexture;
  /** Ворота повторов: на повторе проход пропускается. */
  gate?: FrameGate;
  name?: string;
}
export declare const DEBAND_DEFAULTS: Required<DebandParams>;
/**
 * Дебандинг до сети (Э21): Kodik после x264 приносит полосы на градиентах —
 * небо, дым, тёмные сцены. Сеть ступеньку принимает за край и прорисовывает,
 * дизеринг финального blit от них не спасает — он лишь не добавляет своих.
 * Выход в rgba16float: сглаженный градиент сохраняет точность выше 8 бит.
 */
export declare class Deband implements Anime4KPipeline {
  name: string;
  ready: Promise<void>;
  private device;
  private pipeline?;
  private bindGroup;
  private params;
  private values;
  private launch;
  private outputTexture;
  constructor({ device, inputTexture, gate, name, ...params }: DebandPipelineDescriptor);
  /** Новые параметры; неуказанные остаются прежними. */
  set(params: DebandParams): void;
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}
//#endregion