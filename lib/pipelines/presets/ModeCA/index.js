
import { whenReady } from "../../interfaces.js";
import { FrameGate } from "../../helpers/FrameGate/index.js";
import { ClampStats } from "../../helpers/ClampHighlights/stats.js";
import { Downscale } from "../../helpers/Downscale/index.js";
import { CNNM } from "../../restore/CNNM/index.js";
import { CNNx2M } from "../../upscale/CNNx2M/index.js";
import { DenoiseCNNx2VL } from "../../upscale/DenoiseCNNx2VL/index.js";
import { DenoiseCNNx2M } from "../../upscale/DenoiseCNNx2M/index.js";
import { DenoiseCNNx2L } from "../../upscale/DenoiseCNNx2L/index.js";
import { planModeC } from "../chain.js";
//#region src/pipelines/presets/ModeCA/index.ts
var ModeCA = class {
	pipelines;
	outputTexture;
	/** Все шейдеры цепочки скомпилированы (в фоне); до этого pass() не вызывать. */
	ready;
	/** Ворота повторов (skipUnchanged): счётчики — gate.readStats(). */
	gate;
	/**
	* Constructs a new instance of the preset class.
	*
	* @param {Anime4KPresetPipelineDescriptor} options - An object containing
	* the following properties:
	* @param {GPUDevice} options.device - The GPU device to use for the pipeline.
	* @param {GPUTexture} options.inputTexture - The input texture to process.
	* @param {Dimensions} options.nativeDimensions - The original dimensions of the input texture.
	* @param {Dimensions} options.targetDimensions - The target dimension for the output texture.
	* @param {DenoiseModelSize} [options.denoiseModel='VL'] - Size of the Upscale-Denoise model.
	* @param {CNNPrecision} [options.precision='f32'] - Arithmetic precision of the CNN stages.
	*/
	constructor({ device, inputTexture, nativeDimensions, targetDimensions, denoiseModel = "VL", precision = "f32", skipUnchanged = false, unchangedThreshold = 0 }) {
		this.pipelines = [];
		const chain = planModeC(nativeDimensions, targetDimensions);
		let currentTexture = inputTexture;
		const gate = skipUnchanged ? new FrameGate({
			device,
			inputTexture,
			threshold: unchangedThreshold
		}) : void 0;
		if (gate) this.pipelines.push(gate);
		this.gate = gate;
		const stats = new ClampStats({
			device,
			inputTexture,
			gate
		});
		this.pipelines.push(stats);
		if (chain.upscale1) {
			const Denoise = {
				M: DenoiseCNNx2M,
				L: DenoiseCNNx2L,
				VL: DenoiseCNNx2VL
			}[denoiseModel];
			const upscale1 = new Denoise({
				device,
				inputTexture: currentTexture,
				precision,
				gate
			});
			this.pipelines.push(upscale1);
			currentTexture = upscale1.getOutputTexture();
		}
		if (chain.downscale2) {
			const downscale = new Downscale({
				device,
				inputTexture: currentTexture,
				targetDimensions,
				gate
			});
			this.pipelines.push(downscale);
			currentTexture = downscale.getOutputTexture();
		}
		if (chain.downscale4) {
			const downscale = new Downscale({
				device,
				inputTexture: currentTexture,
				targetDimensions: {
					width: Math.ceil(targetDimensions.width / 2),
					height: Math.ceil(targetDimensions.height / 2)
				},
				gate
			});
			this.pipelines.push(downscale);
			currentTexture = downscale.getOutputTexture();
		}
		const restore = new CNNM({
			device,
			inputTexture: currentTexture,
			precision,
			deRing: chain.upscale2 ? void 0 : stats.getOutputTexture(),
			gate
		});
		this.pipelines.push(restore);
		currentTexture = restore.getOutputTexture();
		if (chain.upscale2) {
			const upscale2 = new CNNx2M({
				device,
				inputTexture: currentTexture,
				precision,
				deRing: stats.getOutputTexture(),
				gate
			});
			this.pipelines.push(upscale2);
			currentTexture = upscale2.getOutputTexture();
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
export { ModeCA };
