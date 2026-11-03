/// <reference types="@webgpu/types" />
import { CNNModelPipelineDescriptor } from "../../interfaces.js";
import { CNN } from "../../helpers/CNN/index.js";
//#region src/pipelines/upscale/DenoiseCNNx2L/index.d.ts
export declare class DenoiseCNNx2L extends CNN {
  constructor({ device, inputTexture, precision, deRing, gate }: CNNModelPipelineDescriptor);
}
//#endregion