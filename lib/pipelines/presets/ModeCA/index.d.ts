/// <reference types="@webgpu/types" />
import { FrameGate } from "../../helpers/FrameGate/index.js";
import { Anime4KPipeline, ModeCPresetPipelineDescriptor } from "../../interfaces.js";
//#region src/pipelines/presets/ModeCA/index.d.ts
export declare class ModeCA implements Anime4KPipeline {
  pipelines: Anime4KPipeline[];
  outputTexture: GPUTexture;
  /** Все шейдеры цепочки скомпилированы (в фоне); до этого pass() не вызывать. */
  ready: Promise<void>;
  /** Ворота повторов (skipUnchanged): счётчики — gate.readStats(). */
  gate?: FrameGate;
  /**
   * Constructs a new instance of the preset class.
   *
   * @param {Anime4KPresetPipelineDescriptor} options - An object containing
   * the following properties:
   * @param {GPUDevice} options.device - The GPU device to use for the pipeline.
   * @param {GPUTexture} options.inputTexture - The input texture to process.
   * @param {Dimensions} options.nativeDimensions - The original dimensions of the input texture.
   * @param {Dimensions} options.targetDimensions - The target dimension for the output texture.
   * @param {DenoiseModelSize} [options.denoiseModel='VL'] - Size of the Upscale-Denoise model.
   * @param {CNNPrecision} [options.precision='f32'] - Arithmetic precision of the CNN stages.
   */
  constructor({ device, inputTexture, nativeDimensions, targetDimensions, denoiseModel, precision, skipUnchanged, unchangedThreshold }: ModeCPresetPipelineDescriptor);
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}
//#endregion