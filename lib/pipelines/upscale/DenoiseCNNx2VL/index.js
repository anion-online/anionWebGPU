
import { CNN } from "../../helpers/CNN/index.js";
import model from "./model.js";
//#region src/pipelines/upscale/DenoiseCNNx2VL/index.ts
var DenoiseCNNx2VL = class extends CNN {
	constructor({ device, inputTexture, precision, deRing, gate }) {
		super({
			device,
			inputTexture,
			model,
			name: "DenoiseCNNx2VL",
			precision,
			deRing,
			gate
		});
	}
};
//#endregion
export { DenoiseCNNx2VL };
