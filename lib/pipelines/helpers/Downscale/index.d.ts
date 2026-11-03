/// <reference types="@webgpu/types" />
import { Launch } from "../FrameGate/index.js";
import { Anime4KPipeline, DownscalePipelineDescriptor } from "../../interfaces.js";
//#region src/pipelines/helpers/Downscale/index.d.ts
/**
 * Уменьшение до `targetDimensions`. По умолчанию — Catmull-Rom с ядром,
 * растянутым на коэффициент уменьшения (без алиасинга); 'bilinear' — выборка
 * в точке, как AutoDownscalePre в mpv и 1.0.0. С `deRing` последний проход
 * сразу выполняет Clamp Highlights.
 */
export declare class Downscale implements Anime4KPipeline {
  outputTexture: GPUTexture;
  steps: {
    pipeline?: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    output: GPUTexture;
    launch: Launch;
  }[];
  ready: Promise<void>;
  /** Группа 1: статистика для deRing() или заглушка. */
  deRing: GPUBindGroup;
  name: string;
  constructor({ device, inputTexture, targetDimensions, filter, deRing, gate, name }: DownscalePipelineDescriptor);
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}
//#endregion