/// <reference types="@webgpu/types" />
import { FrameGate, Launch } from "../FrameGate/index.js";
import { Anime4KPipeline, CNNPrecision } from "../../interfaces.js";
import { CNNModel, CNNStage } from "./model.js";
//#region src/pipelines/helpers/CNN/index.d.ts
export interface CNNPipelineDescriptor {
  device: GPUDevice;
  inputTexture: GPUTexture;
  model: CNNModel;
  name?: string;
  precision?: CNNPrecision;
  /**
   * Статистика ClampStats: финальная стадия сразу выполняет Clamp Highlights
   * (зажим ореолов) — без отдельного прохода в разрешении выхода.
   */
  deRing?: GPUTexture;
  /** Ворота повторов: стадии запускаются косвенно и на повторе пропускаются. */
  gate?: FrameGate;
}
interface CompiledStage {
  /** Появляется, когда createComputePipelineAsync завершится. */
  pipeline?: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  /** Группа 1 финальной стадии — статистика для deRing(). */
  deRing?: GPUBindGroup;
  launch: Launch;
}
/**
 * Исполнитель CNN-модели из conversion/cnn.py: стадия — один compute-проход,
 * поток — один пиксель входа (финальная стадия x2 пишет 2×2 пикселя выхода).
 */
export declare class CNN implements Anime4KPipeline {
  name: string;
  precision: CNNPrecision;
  stages: CompiledStage[];
  ready: Promise<void>;
  outputTexture: GPUTexture;
  private block;
  private width;
  private height;
  constructor({ device, inputTexture, model, name, precision, deRing, gate }: CNNPipelineDescriptor);
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}
//#endregion
export { CNNModel, CNNStage };