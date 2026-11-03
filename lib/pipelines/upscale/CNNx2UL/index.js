
import { Conv2d } from "../../helpers/Conv2d/index.js";
import { DepthToSpace } from "../../helpers/DepthToSpace/index.js";
import { Overlay } from "../../helpers/Overlay/index.js";
import conv2dtf_default from "./shaders/conv2dtf.js";
import conv2dtf1_default from "./shaders/conv2dtf1.js";
import conv2dtf2_default from "./shaders/conv2dtf2.js";
import conv2d1tf_default from "./shaders/conv2d1tf.js";
import conv2d1tf1_default from "./shaders/conv2d1tf1.js";
import conv2d1tf2_default from "./shaders/conv2d1tf2.js";
import conv2d2tf_default from "./shaders/conv2d2tf.js";
import conv2d2tf1_default from "./shaders/conv2d2tf1.js";
import conv2d2tf2_default from "./shaders/conv2d2tf2.js";
import conv2d3tf_default from "./shaders/conv2d3tf.js";
import conv2d3tf1_default from "./shaders/conv2d3tf1.js";
import conv2d3tf2_default from "./shaders/conv2d3tf2.js";
import conv2d4tf_default from "./shaders/conv2d4tf.js";
import conv2d4tf1_default from "./shaders/conv2d4tf1.js";
import conv2d4tf2_default from "./shaders/conv2d4tf2.js";
import conv2d5tf_default from "./shaders/conv2d5tf.js";
import conv2d5tf1_default from "./shaders/conv2d5tf1.js";
import conv2d5tf2_default from "./shaders/conv2d5tf2.js";
import conv2d6tf_default from "./shaders/conv2d6tf.js";
import conv2d6tf1_default from "./shaders/conv2d6tf1.js";
import conv2d6tf2_default from "./shaders/conv2d6tf2.js";
import conv2dlasttf_default from "./shaders/conv2dlasttf.js";
import conv2dlasttf1_default from "./shaders/conv2dlasttf1.js";
import conv2dlasttf2_default from "./shaders/conv2dlasttf2.js";
//#region src/pipelines/upscale/CNNx2UL/index.ts
var CNNx2UL = class {
	/**
	*  [0 - 2] conv2d_tf - conv2d_tf2
	*  [3 - 5] conv2d_1_tf - conv2d_1_tf2
	*  [6 - 8] conv2d_2_tf - conv2d_2_tf2
	*  [9 - 11] conv2d_3_tf - conv2d_3_tf2
	*  [12 - 14] conv2d_4_tf - conv2d_4_tf2
	*  [15 - 17] conv2d_5_tf - conv2d_5_tf2
	*  [18 - 20] conv2d_6_tf - conv2d_6_tf2
	*  [21 - 23] conv2d_last_tf - conv2d_last_tf2
	*/
	pipelines = [];
	/**
	* Creates an instance of CNNx2UL.
	*
	* @param {Object} options - The options for the CNNx2UL pipeline.
	* @param {GPUDevice} options.device - The GPU device to use for
	* creating textures and shader modules.
	* @param {GPUTexture} options.inputTexture - The input texture for the pipeline.
	*/
	constructor({ device, inputTexture }) {
		const shaders = [
			conv2dtf_default,
			conv2dtf1_default,
			conv2dtf2_default,
			conv2d1tf_default,
			conv2d1tf1_default,
			conv2d1tf2_default,
			conv2d2tf_default,
			conv2d2tf1_default,
			conv2d2tf2_default,
			conv2d3tf_default,
			conv2d3tf1_default,
			conv2d3tf2_default,
			conv2d4tf_default,
			conv2d4tf1_default,
			conv2d4tf2_default,
			conv2d5tf_default,
			conv2d5tf1_default,
			conv2d5tf2_default,
			conv2d6tf_default,
			conv2d6tf1_default,
			conv2d6tf2_default,
			conv2dlasttf_default,
			conv2dlasttf1_default,
			conv2dlasttf2_default
		];
		for (let i = 0; i < 3; i += 1) this.pipelines.push(new Conv2d({
			device,
			inputTextures: [inputTexture],
			shaderWGSL: shaders[i],
			name: `conv2d_tf_${i}`
		}));
		const outputTextures = [];
		for (let i = 1; i < 7; i += 1) {
			outputTextures.length = 0;
			outputTextures.push(this.pipelines[3 * (i - 1)].getOutputTexture());
			outputTextures.push(this.pipelines[3 * (i - 1) + 1].getOutputTexture());
			outputTextures.push(this.pipelines[3 * (i - 1) + 2].getOutputTexture());
			for (let j = 0; j < 3; j += 1) this.pipelines.push(new Conv2d({
				device,
				inputTextures: outputTextures,
				shaderWGSL: shaders[3 * i + j],
				name: `conv2d_${i}_tf_${j}`
			}));
		}
		outputTextures.length = 0;
		for (let i = 6; i < this.pipelines.length; i += 1) outputTextures.push(this.pipelines[i].getOutputTexture());
		for (let i = 0; i <= 2; i += 1) this.pipelines.push(new Conv2d({
			device,
			inputTextures: outputTextures,
			shaderWGSL: shaders[21 + i],
			name: `conv2d_last_tf_${i}`
		}));
		this.pipelines.push(new DepthToSpace({
			device,
			inputTextures: [
				this.pipelines[21].getOutputTexture(),
				this.pipelines[22].getOutputTexture(),
				this.pipelines[23].getOutputTexture()
			],
			name: "DepthToSpace"
		}));
		this.pipelines.push(new Overlay({
			device,
			inputTextures: [inputTexture, this.pipelines[24].getOutputTexture()],
			outputTextureSize: [2 * inputTexture.width, 2 * inputTexture.height]
		}));
	}
	updateParam(param, value) {
		throw new Error(`${this.constructor.name} has no param`);
	}
	getOutputTexture() {
		return this.pipelines[this.pipelines.length - 1].getOutputTexture();
	}
	pass(encoder) {
		for (let i = 0; i < this.pipelines.length; i += 1) this.pipelines[i].pass(encoder);
	}
};
//#endregion
export { CNNx2UL };
