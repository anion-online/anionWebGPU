
import { Conv2d } from "../../helpers/Conv2d/index.js";
import { Overlay } from "../../helpers/Overlay/index.js";
import conv2dtf_default from "./shaders/conv2dtf.js";
import conv2d1tf_default from "./shaders/conv2d1tf.js";
import conv2d2tf_default from "./shaders/conv2d2tf.js";
import conv2d3tf_default from "./shaders/conv2d3tf.js";
import conv2d4tf_default from "./shaders/conv2d4tf.js";
import conv2d5tf_default from "./shaders/conv2d5tf.js";
import conv2d6tf_default from "./shaders/conv2d6tf.js";
import output_default from "./shaders/output.js";
//#region src/pipelines/restore/CNNSoftM/index.ts
var CNNSoftM = class {
	/**
	* 7 conv2d pipelines and output
	*/
	pipelines = [];
	/**
	* Creates an instance of CNNUL.
	*
	* @param {Object} options - The options for the CNNM pipeline.
	* @param {GPUDevice} options.device - The GPU device to use for
	* creating textures and shader modules.
	* @param {GPUTexture} options.inputTexture - The input texture for the pipeline.
	*/
	constructor({ device, inputTexture }) {
		const shaders = [
			conv2dtf_default,
			conv2d1tf_default,
			conv2d2tf_default,
			conv2d3tf_default,
			conv2d4tf_default,
			conv2d5tf_default,
			conv2d6tf_default
		];
		this.pipelines.push(new Conv2d({
			device,
			inputTextures: [inputTexture],
			shaderWGSL: shaders[0],
			name: "conv2d_tf"
		}));
		for (let i = 1; i < shaders.length; i += 1) this.pipelines.push(new Conv2d({
			device,
			inputTextures: [this.pipelines[i - 1].getOutputTexture()],
			shaderWGSL: shaders[i],
			name: `conv2d_${i}_tf`
		}));
		const outputTextures = [];
		this.fillOutputTextures(outputTextures, 0, 7);
		this.pipelines.push(new Conv2d({
			device,
			inputTextures: outputTextures,
			shaderWGSL: output_default,
			name: "output"
		}));
		this.pipelines.push(new Overlay({
			device,
			inputTextures: [inputTexture, this.pipelines[this.pipelines.length - 1].getOutputTexture()],
			outputTextureSize: [inputTexture.width, inputTexture.height]
		}));
	}
	fillOutputTextures(outputTextures, from, count) {
		for (let i = from; i < from + count; i += 1) outputTextures.push(this.pipelines[i].getOutputTexture());
	}
	updateParam(param, value) {
		throw new Error("Method not implemented.");
	}
	pass(encoder) {
		for (let i = 0; i < this.pipelines.length; i += 1) this.pipelines[i].pass(encoder);
	}
	getOutputTexture() {
		return this.pipelines[this.pipelines.length - 1].getOutputTexture();
	}
};
//#endregion
export { CNNSoftM };
