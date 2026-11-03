
import { Conv2d } from "../../helpers/Conv2d/index.js";
import { Overlay } from "../../helpers/Overlay/index.js";
import conv2dtf_default from "./shaders/conv2dtf.js";
import conv2dtf1_default from "./shaders/conv2dtf1.js";
import conv2dtf2_default from "./shaders/conv2dtf2.js";
import conv2dtf3_default from "./shaders/conv2dtf3.js";
import conv2dtf4_default from "./shaders/conv2dtf4.js";
import conv2dtf5_default from "./shaders/conv2dtf5.js";
import conv2d1tf_default from "./shaders/conv2d1tf.js";
import conv2d2tf_default from "./shaders/conv2d2tf.js";
import conv2d3tf_default from "./shaders/conv2d3tf.js";
import conv2d3tf1_default from "./shaders/conv2d3tf1.js";
import conv2d3tf2_default from "./shaders/conv2d3tf2.js";
import conv2d3tf3_default from "./shaders/conv2d3tf3.js";
import conv2d3tf4_default from "./shaders/conv2d3tf4.js";
import conv2d3tf5_default from "./shaders/conv2d3tf5.js";
import conv2d4tf_default from "./shaders/conv2d4tf.js";
import conv2d5tf_default from "./shaders/conv2d5tf.js";
import conv2d6tf_default from "./shaders/conv2d6tf.js";
import conv2d6tf1_default from "./shaders/conv2d6tf1.js";
import conv2d6tf2_default from "./shaders/conv2d6tf2.js";
import conv2d6tf3_default from "./shaders/conv2d6tf3.js";
import conv2d6tf4_default from "./shaders/conv2d6tf4.js";
import conv2d6tf5_default from "./shaders/conv2d6tf5.js";
import conv2d7tf_default from "./shaders/conv2d7tf.js";
import conv2d8tf_default from "./shaders/conv2d8tf.js";
import conv2d9tf_default from "./shaders/conv2d9tf.js";
import conv2d9tf1_default from "./shaders/conv2d9tf1.js";
import conv2d9tf2_default from "./shaders/conv2d9tf2.js";
import conv2d9tf3_default from "./shaders/conv2d9tf3.js";
import conv2d9tf4_default from "./shaders/conv2d9tf4.js";
import conv2d9tf5_default from "./shaders/conv2d9tf5.js";
import conv2d10tf_default from "./shaders/conv2d10tf.js";
import conv2d11tf_default from "./shaders/conv2d11tf.js";
import conv2d12tf_default from "./shaders/conv2d12tf.js";
import conv2d12tf1_default from "./shaders/conv2d12tf1.js";
import conv2d12tf2_default from "./shaders/conv2d12tf2.js";
import conv2d12tf3_default from "./shaders/conv2d12tf3.js";
import conv2d12tf4_default from "./shaders/conv2d12tf4.js";
import conv2d12tf5_default from "./shaders/conv2d12tf5.js";
import conv2d13tf_default from "./shaders/conv2d13tf.js";
import conv2d14tf_default from "./shaders/conv2d14tf.js";
import conv2d15tf_default from "./shaders/conv2d15tf.js";
import conv2d15tf1_default from "./shaders/conv2d15tf1.js";
import conv2d15tf2_default from "./shaders/conv2d15tf2.js";
import conv2d15tf3_default from "./shaders/conv2d15tf3.js";
import conv2d15tf4_default from "./shaders/conv2d15tf4.js";
import conv2d15tf5_default from "./shaders/conv2d15tf5.js";
import conv2d16tf_default from "./shaders/conv2d16tf.js";
import conv2d17tf_default from "./shaders/conv2d17tf.js";
import conv2d18tf_default from "./shaders/conv2d18tf.js";
import conv2d18tf1_default from "./shaders/conv2d18tf1.js";
import conv2d18tf2_default from "./shaders/conv2d18tf2.js";
import conv2d18tf3_default from "./shaders/conv2d18tf3.js";
import conv2d18tf4_default from "./shaders/conv2d18tf4.js";
import conv2d18tf5_default from "./shaders/conv2d18tf5.js";
import conv2d19tf_default from "./shaders/conv2d19tf.js";
import conv2d20tf_default from "./shaders/conv2d20tf.js";
import conv2d21tf_default from "./shaders/conv2d21tf.js";
import conv2d21tf1_default from "./shaders/conv2d21tf1.js";
import conv2d21tf2_default from "./shaders/conv2d21tf2.js";
import conv2d21tf3_default from "./shaders/conv2d21tf3.js";
import conv2d21tf4_default from "./shaders/conv2d21tf4.js";
import conv2d21tf5_default from "./shaders/conv2d21tf5.js";
import conv2d22tf_default from "./shaders/conv2d22tf.js";
import conv2d23tf_default from "./shaders/conv2d23tf.js";
import conv2d24tf_default from "./shaders/conv2d24tf.js";
import conv2d24tf1_default from "./shaders/conv2d24tf1.js";
import conv2d24tf2_default from "./shaders/conv2d24tf2.js";
import conv2d24tf3_default from "./shaders/conv2d24tf3.js";
import conv2d24tf4_default from "./shaders/conv2d24tf4.js";
import conv2d24tf5_default from "./shaders/conv2d24tf5.js";
import conv2d25tf_default from "./shaders/conv2d25tf.js";
import conv0ups_default from "./shaders/conv0ups.js";
import conv0ups1_default from "./shaders/conv0ups1.js";
import conv0ups2_default from "./shaders/conv0ups2.js";
import conv0ups3_default from "./shaders/conv0ups3.js";
import conv0ups4_default from "./shaders/conv0ups4.js";
import conv0ups5_default from "./shaders/conv0ups5.js";
import overlayConv1ups_default from "./shaders/overlayConv1ups.js";
import overlayConv1ups1_default from "./shaders/overlayConv1ups1.js";
import overlayConv1ups2_default from "./shaders/overlayConv1ups2.js";
import overlayConv1ups3_default from "./shaders/overlayConv1ups3.js";
import overlayConv1ups4_default from "./shaders/overlayConv1ups4.js";
import overlayConv1ups5_default from "./shaders/overlayConv1ups5.js";
import output_default from "./shaders/output.js";
//#region src/pipelines/upscale/GANx4UUL/index.ts
var GANx4UUL = class {
	/**
	* [0-5] conv2d_tf - conv2d_tf5
	* [6-11] conv2d_3_tf - conv2d_3_tf5
	* [12-17] conv2d_6_tf - conv2d_6_tf5
	* [18-23] conv2d_9_tf - conv2d_9_tf5
	* [24-29] conv2d_12_tf - conv2d_12_tf5
	* [30-35] conv2d_15_tf - conv2d_15_tf5
	* [36-41] conv2d_18_tf - conv2d_18_tf5
	* [42-47] conv2d_21_tf - conv2d_21_tf5
	* [48-53] conv2d_24_tf - conv2d_24_tf5
	*/
	pipelines6 = [];
	/**
	* [0] conv2d_1_tf;
	* [1] conv2d_2_tf;
	* [2] conv2d_4_tf;
	* [3] conv2d_5_tf;
	* [4] conv2d_7_tf;
	* [5] conv2d_8_tf;
	* [6] conv2d_10_tf;
	* [7] conv2d_11_tf;
	* [8] conv2d_13_tf;
	* [9] conv2d_14_tf;
	* [10] conv2d_16_tf;
	* [11] conv2d_17_tf;
	* [12] conv2d_19_tf;
	* [13] conv2d_20_tf;
	* [14] conv2d_22_tf;
	* [15] conv2d_23_tf;
	* [16] conv2d_25_tf;
	*/
	pipelines = [];
	/**
	* [0-5] conv0ups - conv0ups5
	* [6-11] overlayConv1ups - overlayConv1ups5
	*/
	pipelinesUps = [];
	/**
	* Creates an instance of GANx4UUL.
	*
	* @param {Object} options - The options for the GANx4UUL pipeline.
	* @param {GPUDevice} options.device - The GPU device to use for
	* creating textures and shader modules.
	* @param {GPUTexture} options.inputTexture - The input texture for the pipeline.
	*/
	constructor({ device, inputTexture }) {
		const shaders6 = [
			conv2dtf_default,
			conv2dtf1_default,
			conv2dtf2_default,
			conv2dtf3_default,
			conv2dtf4_default,
			conv2dtf5_default,
			conv2d3tf_default,
			conv2d3tf1_default,
			conv2d3tf2_default,
			conv2d3tf3_default,
			conv2d3tf4_default,
			conv2d3tf5_default,
			conv2d6tf_default,
			conv2d6tf1_default,
			conv2d6tf2_default,
			conv2d6tf3_default,
			conv2d6tf4_default,
			conv2d6tf5_default,
			conv2d9tf_default,
			conv2d9tf1_default,
			conv2d9tf2_default,
			conv2d9tf3_default,
			conv2d9tf4_default,
			conv2d9tf5_default,
			conv2d12tf_default,
			conv2d12tf1_default,
			conv2d12tf2_default,
			conv2d12tf3_default,
			conv2d12tf4_default,
			conv2d12tf5_default,
			conv2d15tf_default,
			conv2d15tf1_default,
			conv2d15tf2_default,
			conv2d15tf3_default,
			conv2d15tf4_default,
			conv2d15tf5_default,
			conv2d18tf_default,
			conv2d18tf1_default,
			conv2d18tf2_default,
			conv2d18tf3_default,
			conv2d18tf4_default,
			conv2d18tf5_default,
			conv2d21tf_default,
			conv2d21tf1_default,
			conv2d21tf2_default,
			conv2d21tf3_default,
			conv2d21tf4_default,
			conv2d21tf5_default,
			conv2d24tf_default,
			conv2d24tf1_default,
			conv2d24tf2_default,
			conv2d24tf3_default,
			conv2d24tf4_default,
			conv2d24tf5_default
		];
		const shaders = [
			conv2d1tf_default,
			conv2d2tf_default,
			conv2d4tf_default,
			conv2d5tf_default,
			conv2d7tf_default,
			conv2d8tf_default,
			conv2d10tf_default,
			conv2d11tf_default,
			conv2d13tf_default,
			conv2d14tf_default,
			conv2d16tf_default,
			conv2d17tf_default,
			conv2d19tf_default,
			conv2d20tf_default,
			conv2d22tf_default,
			conv2d23tf_default,
			conv2d25tf_default
		];
		const shadersUps = [
			conv0ups_default,
			conv0ups1_default,
			conv0ups2_default,
			conv0ups3_default,
			conv0ups4_default,
			conv0ups5_default,
			overlayConv1ups_default,
			overlayConv1ups1_default,
			overlayConv1ups2_default,
			overlayConv1ups3_default,
			overlayConv1ups4_default,
			overlayConv1ups5_default
		];
		let len = this.pipelines6.length;
		for (let i = 0; i < 6; i += 1) this.pipelines6.push(new Conv2d({
			device,
			inputTextures: [inputTexture],
			shaderWGSL: shaders6[i],
			name: `conv2d_tf_${i}`
		}));
		const outputTextures = [];
		for (let i = 0; i < 8; i += 1) {
			outputTextures.length = 0;
			this.fillOutputTexturesFromPipeline6(outputTextures, len, 6);
			for (let j = 0; j < 2; j += 1) this.pipelines.push(new Conv2d({
				device,
				inputTextures: outputTextures,
				shaderWGSL: shaders[j + 2 * i],
				name: `conv2d_${j + 3 * i + 1}_tf`
			}));
			this.addOutputTexturesFromPipeline(outputTextures);
			len = this.pipelines6.length;
			for (let j = 0; j < 6; j += 1) this.pipelines6.push(new Conv2d({
				device,
				inputTextures: outputTextures,
				shaderWGSL: shaders6[len + j],
				name: `conv2d_${3 * (i + 1)}_tf${j}`
			}));
		}
		outputTextures.length = 0;
		this.fillOutputTexturesFromPipeline6(outputTextures, 48, 6);
		this.pipelines.push(new Conv2d({
			device,
			inputTextures: outputTextures,
			shaderWGSL: shaders[shaders.length - 1],
			name: "conv2d_25_tf"
		}));
		outputTextures.push(this.pipelines[15].getOutputTexture());
		for (let i = 0; i < 15; i += 1) if (i % 2 === 0) outputTextures.push(this.pipelines[i].getOutputTexture());
		outputTextures.push(this.pipelines[this.pipelines.length - 1].getOutputTexture());
		for (let i = 0; i < 6; i += 1) this.pipelinesUps.push(new Conv2d({
			device,
			inputTextures: outputTextures,
			shaderWGSL: shadersUps[i],
			name: `conv0ups${i}`
		}));
		outputTextures.length = 0;
		for (let i = 0; i < this.pipelinesUps.length; i += 1) outputTextures.push(this.pipelinesUps[i].getOutputTexture());
		for (let i = 0; i < 6; i += 1) this.pipelinesUps.push(new Overlay({
			device,
			inputTextures: outputTextures,
			outputTextureSize: [4 * inputTexture.width, 4 * inputTexture.height],
			fragmentWGSL: shadersUps[i + 6],
			name: `overlay_conv1ups${i}`
		}));
		outputTextures.length = 0;
		for (let i = 6; i < this.pipelinesUps.length; i += 1) outputTextures.push(this.pipelinesUps[i].getOutputTexture());
		this.pipelinesUps.push(new Conv2d({
			device,
			inputTextures: outputTextures,
			shaderWGSL: output_default,
			name: "output"
		}));
		this.pipelinesUps.push(new Overlay({
			device,
			inputTextures: [this.pipelinesUps[this.pipelinesUps.length - 1].getOutputTexture(), inputTexture],
			outputTextureSize: [4 * inputTexture.width, 4 * inputTexture.height]
		}));
	}
	fillOutputTexturesFromPipeline6(outputTextures, from, count) {
		for (let i = from; i < from + count; i += 1) outputTextures.push(this.pipelines6[i].getOutputTexture());
	}
	addOutputTexturesFromPipeline(outputTextures) {
		const len = this.pipelines.length;
		outputTextures.push(this.pipelines[len - 1].getOutputTexture());
		for (let i = 0; i < len; i += 1) if (i % 2 === 0) outputTextures.push(this.pipelines[i].getOutputTexture());
	}
	updateParam(param, value) {
		throw new Error("Method not implemented.");
	}
	pass(encoder) {
		for (let i = 0; i < 9; i += 1) {
			for (let j = 0; j < 6; j += 1) this.pipelines6[6 * i + j].pass(encoder);
			if (i !== 8) {
				this.pipelines[2 * i].pass(encoder);
				this.pipelines[2 * i + 1].pass(encoder);
			}
		}
		this.pipelines[16].pass(encoder);
		for (let i = 0; i < this.pipelinesUps.length; i += 1) this.pipelinesUps[i].pass(encoder);
	}
	getOutputTexture() {
		return this.pipelinesUps[this.pipelinesUps.length - 1].getOutputTexture();
	}
};
//#endregion
export { GANx4UUL };
