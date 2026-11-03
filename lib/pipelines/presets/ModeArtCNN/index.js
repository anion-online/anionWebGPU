
import { whenReady } from "../../interfaces.js";
import { Downscale } from "../../helpers/Downscale/index.js";
import { ArtCNNC4F16DS } from "../../artcnn/ArtCNNC4F16DS/index.js";
import { ArtCNNC4F32DS } from "../../artcnn/ArtCNNC4F32DS/index.js";
//#region src/pipelines/presets/ModeArtCNN/index.ts
/**
* ArtCNN DS (denoise + sharpen) ×2 по яркости, затем уменьшение до цели,
* если она меньше 2× (Catmull-Rom). При цели ≤ 1.2× исходника не делает
* ничего — выход равен входу, как у ModeC.
*/
var ModeArtCNN = class {
	pipelines = [];
	outputTexture;
	/** Все шейдеры цепочки скомпилированы (в фоне); до этого pass() не вызывать. */
	ready;
	constructor({ device, inputTexture, nativeDimensions, targetDimensions, model = "C4F16", precision = "f32" }) {
		let currentTexture = inputTexture;
		if (targetDimensions.width > 1.2 * nativeDimensions.width && targetDimensions.height > 1.2 * nativeDimensions.height) {
			const upscale = new (model === "C4F32" ? ArtCNNC4F32DS : ArtCNNC4F16DS)({
				device,
				inputTexture: currentTexture,
				precision
			});
			this.pipelines.push(upscale);
			currentTexture = upscale.getOutputTexture();
			if (targetDimensions.width < 2 * nativeDimensions.width && targetDimensions.height < 2 * nativeDimensions.height) {
				const downscale = new Downscale({
					device,
					inputTexture: currentTexture,
					targetDimensions
				});
				this.pipelines.push(downscale);
				currentTexture = downscale.getOutputTexture();
			}
		}
		this.outputTexture = currentTexture;
		this.ready = whenReady(this.pipelines);
	}
	updateParam(param, value) {
		throw new Error("Preset has no param");
	}
	pass(encoder) {
		this.pipelines.forEach((pipeline) => pipeline.pass(encoder));
	}
	getOutputTexture() {
		return this.outputTexture;
	}
};
//#endregion
export { ModeArtCNN };
