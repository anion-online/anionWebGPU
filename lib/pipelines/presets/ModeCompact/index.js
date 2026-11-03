
import { whenReady } from "../../interfaces.js";
import { FrameGate } from "../../helpers/FrameGate/index.js";
import { Deband } from "../../denoise/Deband/index.js";
import { Downscale } from "../../helpers/Downscale/index.js";
import { CNNM } from "../../restore/CNNM/index.js";
import { CompactSR } from "../../compact/index.js";
//#region src/pipelines/presets/ModeCompact/index.ts
/**
* «Детали»: SRVGGNetCompact ×2 (AnimeJaNai V2 SuperUltraCompact), затем
* уменьшение до цели, если она меньше 2× (Catmull-Rom). При цели ≤ 1.2×
* исходника не делает ничего — как ModeArtCNN.
*/
var ModeCompact = class {
	pipelines = [];
	outputTexture;
	ready;
	/** Ворота повторов (skipUnchanged): счётчики — gate.readStats(). */
	gate;
	/** Дебандинг (deband): параметры меняются на ходу — deband.set(). */
	deband;
	constructor({ device, inputTexture, nativeDimensions, targetDimensions, model, kernel, skipUnchanged = false, unchangedThreshold = 0, chromaSigma = 2, chromaThreshold = 0, deband, restore }) {
		let currentTexture = inputTexture;
		if (targetDimensions.width > 1.2 * nativeDimensions.width && targetDimensions.height > 1.2 * nativeDimensions.height) {
			const gate = skipUnchanged ? new FrameGate({
				device,
				inputTexture,
				threshold: unchangedThreshold
			}) : void 0;
			if (gate) this.pipelines.push(gate);
			this.gate = gate;
			if (deband) {
				this.deband = new Deband({
					device,
					inputTexture: currentTexture,
					gate,
					...deband === true ? {} : deband
				});
				this.pipelines.push(this.deband);
				currentTexture = this.deband.getOutputTexture();
			}
			if (restore === "before") {
				const cnnm = new CNNM({
					device,
					inputTexture: currentTexture,
					gate
				});
				this.pipelines.push(cnnm);
				currentTexture = cnnm.getOutputTexture();
			}
			const upscale = new CompactSR({
				device,
				inputTexture: currentTexture,
				model,
				kernel,
				gate,
				chromaSigma,
				chromaThreshold
			});
			this.pipelines.push(upscale);
			currentTexture = upscale.getOutputTexture();
			if (targetDimensions.width < 2 * nativeDimensions.width && targetDimensions.height < 2 * nativeDimensions.height) {
				const downscale = new Downscale({
					device,
					inputTexture: currentTexture,
					targetDimensions,
					gate
				});
				this.pipelines.push(downscale);
				currentTexture = downscale.getOutputTexture();
			}
			if (restore === "after") {
				const cnnm = new CNNM({
					device,
					inputTexture: currentTexture,
					gate
				});
				this.pipelines.push(cnnm);
				currentTexture = cnnm.getOutputTexture();
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
export { ModeCompact };
