/// <reference types="@webgpu/types" />
import { Anime4KPipeline, ModeArtCNNPresetPipelineDescriptor } from "../../interfaces.js";
//#region src/pipelines/presets/ModeArtCNN/index.d.ts
/**
 * ArtCNN DS (denoise + sharpen) ×2 по яркости, затем уменьшение до цели,
 * если она меньше 2× (Catmull-Rom). При цели ≤ 1.2× исходника не делает
 * ничего — выход равен входу, как у ModeC.
 */
export declare class ModeArtCNN implements Anime4KPipeline {
  pipelines: Anime4KPipeline[];
  outputTexture: GPUTexture;
  /** Все шейдеры цепочки скомпилированы (в фоне); до этого pass() не вызывать. */
  ready: Promise<void>;
  constructor({ device, inputTexture, nativeDimensions, targetDimensions, model, precision }: ModeArtCNNPresetPipelineDescriptor);
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}
//#endregion