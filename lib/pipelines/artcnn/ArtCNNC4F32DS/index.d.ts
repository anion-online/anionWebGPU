/// <reference types="@webgpu/types" />
import { CNNModelPipelineDescriptor } from "../../interfaces.js";
import { CNN } from "../../helpers/CNN/index.js";
//#region src/pipelines/artcnn/ArtCNNC4F32DS/index.d.ts
export declare class ArtCNNC4F32DS extends CNN {
  constructor({ device, inputTexture, precision, deRing }: CNNModelPipelineDescriptor);
}
//#endregion