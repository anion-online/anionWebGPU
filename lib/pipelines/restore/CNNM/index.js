
import { CNN } from "../../helpers/CNN/index.js";
import model from "./model.js";
//#region src/pipelines/restore/CNNM/index.ts
var CNNM = class extends CNN {
	constructor({ device, inputTexture, precision, deRing, gate }) {
		super({
			device,
			inputTexture,
			model,
			name: "CNNM",
			precision,
			deRing,
			gate
		});
	}
};
//#endregion
export { CNNM };
