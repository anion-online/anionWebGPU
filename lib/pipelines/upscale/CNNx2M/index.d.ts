/// <reference types="@webgpu/types" />
import { CNNModelPipelineDescriptor } from "../../interfaces.js";
import { CNN } from "../../helpers/CNN/index.js";
//#region src/pipelines/upscale/CNNx2M/index.d.ts
export declare class CNNx2M extends CNN {
  constructor({ device, inputTexture, precision, deRing, gate }: CNNModelPipelineDescriptor);
}
//#endregion