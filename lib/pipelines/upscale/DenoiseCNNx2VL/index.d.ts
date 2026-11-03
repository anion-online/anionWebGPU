/// <reference types="@webgpu/types" />
import { CNNModelPipelineDescriptor } from "../../interfaces.js";
import { CNN } from "../../helpers/CNN/index.js";
//#region src/pipelines/upscale/DenoiseCNNx2VL/index.d.ts
export declare class DenoiseCNNx2VL extends CNN {
  constructor({ device, inputTexture, precision, deRing, gate }: CNNModelPipelineDescriptor);
}
//#endregion