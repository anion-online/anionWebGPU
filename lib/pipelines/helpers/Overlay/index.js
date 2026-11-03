
import vertex_default from "./shaders/vertex.js";
import overlay2_default from "./shaders/overlay2.js";
//#region src/pipelines/helpers/Overlay/index.ts
/**
* Render Pipeline:
*      Takes in n input textures output texture.
*/
var Overlay = class {
	outputTexture;
	pipeline;
	bindGroup;
	name;
	/**
	* Creates an instance of Overlay.
	*
	* @param {Object} options - The options for the Overlay pipeline.
	* @param {GPUDevice} options.device - The GPU device to use for creating
	*  textures and shader modules.
	* @param {Array<GPUTexture>} options.inputTextures - The input textures for the pipeline.
	* All input textures must have the same dimensions.
	* @param {Array<number>} options.outputTextureSize - The size of the output texture.
	* @param {string} [options.fragmentWGSL=overlay2WGSL] - The fragment shader code in WGSL format.
	* Defaults to 'overlay2WGSL' (overlay 2 textures).
	* @param {string} [options.name='overlay'] - The name of the pipeline. Defaults to 'overlay'.
	*
	* @throws {Error} Will throw an error if the shader is not defined.
	*/
	constructor({ device, inputTextures, outputTextureSize, fragmentWGSL = overlay2_default, name = "overlay" }) {
		const inputLength = inputTextures.length;
		this.name = name;
		if (fragmentWGSL === void 0) throw Error(`${name}: shader not defined.`);
		this.outputTexture = device.createTexture({
			label: `${name}: output texture`,
			size: [
				outputTextureSize[0],
				outputTextureSize[1],
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
		});
		const vertexModule = device.createShaderModule({
			label: `${name}: vertex module`,
			code: vertex_default
		});
		const fragmentModule = device.createShaderModule({
			label: `${name}: fragment module`,
			code: fragmentWGSL
		});
		const bindGroupLayoutEntries = [];
		bindGroupLayoutEntries.push({
			binding: 0,
			visibility: GPUShaderStage.FRAGMENT,
			sampler: {}
		});
		for (let i = 1; i <= inputLength; i += 1) bindGroupLayoutEntries.push({
			binding: i,
			visibility: GPUShaderStage.FRAGMENT,
			texture: {}
		});
		const bindGroupLayout = device.createBindGroupLayout({
			label: `${name}: bind group layout`,
			entries: bindGroupLayoutEntries
		});
		const pipelineLayout = device.createPipelineLayout({
			label: `${name}: pipeline layout`,
			bindGroupLayouts: [bindGroupLayout]
		});
		this.pipeline = device.createRenderPipeline({
			layout: pipelineLayout,
			vertex: {
				module: vertexModule,
				entryPoint: "vert_main"
			},
			fragment: {
				module: fragmentModule,
				entryPoint: "main",
				targets: [{ format: "rgba16float" }]
			},
			primitive: { topology: "triangle-list" }
		});
		const sampler = device.createSampler({
			magFilter: "linear",
			minFilter: "linear"
		});
		const bindGroupEntries = [];
		bindGroupEntries.push({
			binding: 0,
			resource: sampler
		});
		for (let i = 1; i <= inputLength; i += 1) bindGroupEntries.push({
			binding: i,
			resource: inputTextures[i - 1].createView()
		});
		this.bindGroup = device.createBindGroup({
			label: `${name}: bind group`,
			layout: bindGroupLayout,
			entries: bindGroupEntries
		});
	}
	updateParam(param, value) {
		throw new Error(`${this.constructor.name} has no param`);
	}
	pass(encoder) {
		const bilinearPass = encoder.beginRenderPass({ colorAttachments: [{
			view: this.outputTexture.createView(),
			clearValue: {
				r: 0,
				g: 0,
				b: 0,
				a: 1
			},
			loadOp: "clear",
			storeOp: "store"
		}] });
		bilinearPass.setPipeline(this.pipeline);
		bilinearPass.setBindGroup(0, this.bindGroup);
		bilinearPass.draw(6);
		bilinearPass.end();
	}
	getOutputTexture() {
		return this.outputTexture;
	}
};
//#endregion
export { Overlay };
