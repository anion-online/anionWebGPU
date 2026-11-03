/// <reference types="@webgpu/types" />
import { Anime4KPipeline, ClampHighlightsPipelineDescriptor } from "../../interfaces.js";
import { ClampStats, ClampStatsPipelineDescriptor, DeRingEpilogue } from "./stats.js";
//#region src/pipelines/helpers/ClampHighlights/index.d.ts
/**
 * Anime4K Clamp Highlights (de-ring).
 *
 * Статистика (максимум яркости 5×5) снимается с исходника `statsTexture`, а
 * зажим применяется к `inputTexture` — результату всей цепочки, в его
 * разрешении. Поэтому звено ставится последним. В 1.0.0 оба шага шли по одному
 * кадру в начале цепочки, и зажим ничего не делал: окно включает сам пиксель.
 */
export declare class ClampHighlights implements Anime4KPipeline {
  name: string;
  pipelines: {
    statsPipeline: GPUComputePipeline;
    clampPipeline: GPUComputePipeline;
  };
  bindGroups: {
    statsBindGroup: GPUBindGroup;
    clampBindGroup: GPUBindGroup;
  };
  statsTexture: GPUTexture;
  outputTexture: GPUTexture;
  constructor({ device, inputTexture, statsTexture, name }: ClampHighlightsPipelineDescriptor);
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}
//#endregion
export { ClampStats, ClampStatsPipelineDescriptor, DeRingEpilogue };