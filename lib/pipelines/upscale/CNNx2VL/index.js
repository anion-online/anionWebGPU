
import { Conv2d } from "../../helpers/Conv2d/index.js";
import { DepthToSpace } from "../../helpers/DepthToSpace/index.js";
import { Overlay } from "../../helpers/Overlay/index.js";
import conv2dtf_default from "./shaders/conv2dtf.js";
import conv2dtf1_default from "./shaders/conv2dtf1.js";
import conv2d1tf_default from "./shaders/conv2d1tf.js";
import conv2d1tf1_default from "./shaders/conv2d1tf1.js";
import conv2d2tf_default from "./shaders/conv2d2tf.js";
import conv2d2tf1_default from "./shaders/conv2d2tf1.js";
import conv2d3tf_default from "./shaders/conv2d3tf.js";
import conv2d3tf1_default from "./shaders/conv2d3tf1.js";
import conv2d4tf_default from "./shaders/conv2d4tf.js";
import conv2d4tf1_default from "./shaders/conv2d4tf1.js";
import conv2d5tf_default from "./shaders/conv2d5tf.js";
import conv2d5tf1_default from "./shaders/conv2d5tf1.js";
import conv2d6tf_default from "./shaders/conv2d6tf.js";
import conv2d6tf1_default from "./shaders/conv2d6tf1.js";
import conv2dlasttf_default from "./shaders/conv2dlasttf.js";
import conv2dlasttf1_default from "./shaders/conv2dlasttf1.js";
import conv2dlasttf2_default from "./shaders/conv2dlasttf2.js";
//#region src/pipelines/upscale/CNNx2VL/index.ts
var CNNx2VL = class {
	pipelines = [];
	/**
	* Creates an instance of CNNSoftVL.
	*
	* @param {Object} options - The options for the CNNSoftVL pipeline.
	* @param {GPUDevice} options.device - The GPU device to use for
	* creating textures and shader modules.
	* @param {GPUTexture} options.inputTexture - The input texture for the pipeline.
	*/
	constructor({ device, inputTexture }) {
		const shaders = [
			conv2dtf_default,
			conv2dtf1_default,
			conv2d1tf_default,
			conv2d1tf1_default,
			conv2d2tf_default,
			conv2d2tf1_default,
			conv2d3tf_default,
			conv2d3tf1_default,
			conv2d4tf_default,
			conv2d4tf1_default,
			conv2d5tf_default,
			conv2d5tf1_default,
			conv2d6tf_default,
			conv2d6tf1_default,
			conv2dlasttf_default,
			conv2dlasttf1_default,
			conv2dlasttf2_default
		];
		this.pushPipeline(device, [inputTexture], shaders[0], "conv2d_tf");
		this.pushPipeline(device, [inputTexture], shaders[1], "conv2d_tf_1");
		const inputTextures = [];
		for (let i = 1; i < 7; i += 1) {
			inputTextures.length = 0;
			this.fillOutputTextures(inputTextures, 2 * (i - 1), 2);
			this.pushPipeline(device, inputTextures, shaders[2 * i], `conv2d_${i}_tf`);
			this.pushPipeline(device, inputTextures, shaders[2 * i + 1], `conv2d_${i}_tf_1`);
		}
		inputTextures.length = 0;
		this.fillOutputTextures(inputTextures, 0, this.pipelines.length);
		const len = shaders.length;
		for (let i = 0; i < 3; i += 1) this.pushPipeline(device, inputTextures, shaders[len - 3 + i], `conv2d_last_tf_${i}`);
		inputTextures.length = 0;
		this.fillOutputTextures(inputTextures, this.pipelines.length - 3, 3);
		this.pipelines.push(new DepthToSpace({
			device,
			inputTextures,
			name: "DepthToSpace"
		}));
		this.pipelines.push(new Overlay({
			device,
			inputTextures: [inputTexture, this.getOutputTexture()],
			outputTextureSize: [2 * inputTexture.width, 2 * inputTexture.height]
		}));
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
	pushPipeline(device, inputTexture, shaderWGSL, name) {
		this.pipelines.push(new Conv2d({
			device,
			inputTextures: inputTexture,
			shaderWGSL,
			name
		}));
	}
	fillOutputTextures(outputTextures, from, count) {
		for (let i = from; i < from + count; i += 1) outputTextures.push(this.pipelines[i].getOutputTexture());
	}
};
//#endregion
export { CNNx2VL };
