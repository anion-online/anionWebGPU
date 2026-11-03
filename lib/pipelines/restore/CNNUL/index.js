
import { Conv2d } from "../../helpers/Conv2d/index.js";
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
import conv2d7tf_default from "./shaders/conv2d7tf.js";
import conv2d7tf1_default from "./shaders/conv2d7tf1.js";
import conv2d7tf2_default from "./shaders/conv2d7tf2.js";
import output_default from "./shaders/output.js";
//#region src/pipelines/restore/CNNUL/index.ts
var CNNUL = class {
	/**
	*  [0 - 2] conv2d_tf - conv2d_tf2
	*  [3 - 5] conv2d_1_tf - conv2d_1_tf2
	*  [6 - 8] conv2d_2_tf - conv2d_2_tf2
	*  [9 - 11] conv2d_3_tf - conv2d_3_tf2
	*  [12 - 14] conv2d_4_tf - conv2d_4_tf2
	*  [15 - 17] conv2d_5_tf - conv2d_5_tf2
	*  [18 - 20] conv2d_6_tf - conv2d_6_tf2
	*  [21 - 23] conv2d_7_tf - conv2d_7_tf2
	*  [24] output
	*/
	pipelines = [];
	/**
	* Creates an instance of CNNUL.
	*
	* @param {Object} options - The options for the CNNUL pipeline.
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
			conv2d7tf_default,
			conv2d7tf1_default,
			conv2d7tf2_default
		];
		for (let i = 0; i < 3; i += 1) this.pipelines.push(new Conv2d({
			device,
			inputTextures: [inputTexture],
			shaderWGSL: shaders[i],
			name: `conv2d_tf_${i}`
		}));
		const outputTextures = [];
		for (let i = 1; i <= 7; i += 1) {
			outputTextures.length = 0;
			this.fillOutputTextures(outputTextures, (i - 1) * 3, 3);
			for (let j = 0; j < 3; j += 1) this.pipelines.push(new Conv2d({
				device,
				inputTextures: outputTextures,
				shaderWGSL: shaders[i * 3 + j],
				name: `conv2d_${i}_tf_${j}`
			}));
		}
		outputTextures.length = 0;
		this.fillOutputTextures(outputTextures, 9, 15);
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
export { CNNUL };
