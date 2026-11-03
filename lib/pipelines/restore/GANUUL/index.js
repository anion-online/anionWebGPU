
import { Conv2d } from "../../helpers/Conv2d/index.js";
import { Overlay } from "../../helpers/Overlay/index.js";
import conv2dtf_default from "./shaders/conv2dtf.js";
import conv2dtf1_default from "./shaders/conv2dtf1.js";
import conv2d1tf_default from "./shaders/conv2d1tf.js";
import conv2d1tf1_default from "./shaders/conv2d1tf1.js";
import conv2d1tf2_default from "./shaders/conv2d1tf2.js";
import conv2d2tf_default from "./shaders/conv2d2tf.js";
import conv2d2tf1_default from "./shaders/conv2d2tf1.js";
import conv2d3tf_default from "./shaders/conv2d3tf.js";
import conv2d3tf1_default from "./shaders/conv2d3tf1.js";
import conv2d3tf2_default from "./shaders/conv2d3tf2.js";
import conv2d4tf_default from "./shaders/conv2d4tf.js";
import conv2d4tf1_default from "./shaders/conv2d4tf1.js";
import conv2d5tf_default from "./shaders/conv2d5tf.js";
import conv2d5tf1_default from "./shaders/conv2d5tf1.js";
import conv2d5tf2_default from "./shaders/conv2d5tf2.js";
import conv2d6tf_default from "./shaders/conv2d6tf.js";
import conv2d6tf1_default from "./shaders/conv2d6tf1.js";
import output_default from "./shaders/output.js";
//#region src/pipelines/restore/GANUUL/index.ts
var GANUUL = class {
	/**
	* pipelines:
	*  [0] conv2d_tf
	*  [1] conv2d_tf1
	*  [2] conv2d_1_tf
	*  [3] conv2d_1_tf1
	*  [4] conv2d_1_tf2
	*  [5] conv2d_2_tf
	*  [6] conv2d_2_tf1
	*  [7] conv2d_3_tf
	*  [8] conv2d_3_tf1
	*  [9] conv2d_3_tf2
	*  [10] conv2d_4_tf
	*  [11] conv2d_4_tf1
	*  [12] conv2d_5_tf
	*  [13] conv2d_5_tf1
	*  [14] conv2d_5_tf2
	*  [15] conv2d_6_tf
	*  [16] conv2d_6_tf1
	*  [17] output
	*/
	pipelines = [];
	/**
	* Creates an instance of GANUUL.
	*
	* @param {Object} options - The options for the GANUUL pipeline.
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
			conv2d1tf2_default,
			conv2d2tf_default,
			conv2d2tf1_default,
			conv2d3tf_default,
			conv2d3tf1_default,
			conv2d3tf2_default,
			conv2d4tf_default,
			conv2d4tf1_default,
			conv2d5tf_default,
			conv2d5tf1_default,
			conv2d5tf2_default,
			conv2d6tf_default,
			conv2d6tf1_default,
			output_default
		];
		this.pipelines.push(new Conv2d({
			device,
			inputTextures: [inputTexture],
			shaderWGSL: shaders[0],
			name: "conv2d_tf"
		}));
		this.pipelines.push(new Conv2d({
			device,
			inputTextures: [inputTexture],
			shaderWGSL: shaders[1],
			name: "conv2d_tf1"
		}));
		const outputTextures = [];
		this.fillOutputTextures(outputTextures, 0, 2);
		for (let i = 0; i < 3; i += 1) this.pipelines.push(new Conv2d({
			device,
			inputTextures: outputTextures,
			shaderWGSL: shaders[i + 2],
			name: `conv2d_1_tf_${i}`
		}));
		outputTextures.length = 0;
		this.fillOutputTextures(outputTextures, 2, 3);
		for (let i = 0; i < 2; i += 1) this.pipelines.push(new Conv2d({
			device,
			inputTextures: outputTextures,
			shaderWGSL: shaders[i + 5],
			name: `conv2d_2_tf_${i}`
		}));
		outputTextures.length = 0;
		this.fillOutputTextures(outputTextures, 0, 2);
		this.fillOutputTextures(outputTextures, 5, 2);
		for (let i = 0; i < 3; i += 1) this.pipelines.push(new Conv2d({
			device,
			inputTextures: outputTextures,
			shaderWGSL: shaders[i + 7],
			name: `conv2d_3_tf_${i}`
		}));
		outputTextures.length = 0;
		this.fillOutputTextures(outputTextures, 7, 3);
		for (let i = 0; i < 2; i += 1) this.pipelines.push(new Conv2d({
			device,
			inputTextures: outputTextures,
			shaderWGSL: shaders[i + 10],
			name: `conv2d_4_tf_${i}`
		}));
		outputTextures.length = 0;
		this.fillOutputTextures(outputTextures, 5, 2);
		this.fillOutputTextures(outputTextures, 10, 2);
		for (let i = 0; i < 3; i += 1) this.pipelines.push(new Conv2d({
			device,
			inputTextures: outputTextures,
			shaderWGSL: shaders[i + 12],
			name: `conv2d_5_tf_${i}`
		}));
		outputTextures.length = 0;
		this.fillOutputTextures(outputTextures, 12, 3);
		for (let i = 0; i < 2; i += 1) this.pipelines.push(new Conv2d({
			device,
			inputTextures: outputTextures,
			shaderWGSL: shaders[i + 15],
			name: `conv2d_6_tf_${i}`
		}));
		outputTextures.length = 0;
		this.fillOutputTextures(outputTextures, 10, 2);
		this.fillOutputTextures(outputTextures, 15, 2);
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
export { GANUUL };
