/// <reference types="@webgpu/types" />
import { CNNModelPipelineDescriptor } from "../../interfaces.js";
import { CNN } from "../../helpers/CNN/index.js";
//#region src/pipelines/restore/CNNM/index.d.ts
export declare class CNNM extends CNN {
  constructor({ device, inputTexture, precision, deRing, gate }: CNNModelPipelineDescriptor);
}
//#endregion