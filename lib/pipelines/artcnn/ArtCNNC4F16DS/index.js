
import { CNN } from "../../helpers/CNN/index.js";
import model from "./model.js";
//#region src/pipelines/artcnn/ArtCNNC4F16DS/index.ts
var ArtCNNC4F16DS = class extends CNN {
	constructor({ device, inputTexture, precision, deRing }) {
		super({
			device,
			inputTexture,
			model,
			name: "ArtCNNC4F16DS",
			precision,
			deRing
		});
	}
};
//#endregion
export { ArtCNNC4F16DS };
