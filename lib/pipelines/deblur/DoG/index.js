
import lumination_default from "./shaders/lumination.js";
import deblurDoGX_default from "./shaders/deblurDoGX.js";
import deblurDoGY_default from "./shaders/deblurDoGY.js";
import deblurDoGApply_default from "./shaders/deblurDoGApply.js";
//#region src/pipelines/deblur/DoG/index.ts
var DoG = class {
	textures;
	modules;
	bindGroupLayouts;
	bindGroups;
	pipelineLayouts;
	pipelines;
	strengthBuffer;
	inputTexWidth;
	inputTexHeight;
	inputTexture;
	device;
	constructor({ device, inputTexture }) {
		this.device = device;
		this.inputTexWidth = inputTexture.width;
		this.inputTexHeight = inputTexture.height;
		this.inputTexture = inputTexture;
		const luminationBindGroupLayout = device.createBindGroupLayout({
			label: "lumination Bind Group Layout",
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.COMPUTE,
				texture: {}
			}, {
				binding: 1,
				visibility: GPUShaderStage.COMPUTE,
				storageTexture: {
					access: "write-only",
					format: "rgba16float"
				}
			}]
		});
		const luminationModule = device.createShaderModule({
			label: "lumination shader",
			code: lumination_default
		});
		const luminationPipelineLayout = device.createPipelineLayout({
			label: "lumination pipeline layout",
			bindGroupLayouts: [luminationBindGroupLayout]
		});
		const luminationPipeline = device.createComputePipeline({
			label: "lumination pipeline",
			layout: luminationPipelineLayout,
			compute: {
				module: luminationModule,
				entryPoint: "computeMain"
			}
		});
		const luminationTexture = device.createTexture({
			size: [
				this.inputTexWidth,
				this.inputTexHeight,
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		const deblurDoGXBindGroupLayout = device.createBindGroupLayout({
			label: "deblurDoGX Bind Group Layout",
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.COMPUTE,
				texture: {}
			}, {
				binding: 1,
				visibility: GPUShaderStage.COMPUTE,
				storageTexture: {
					access: "write-only",
					format: "rgba16float"
				}
			}]
		});
		const deblurDoGXModule = device.createShaderModule({
			label: "deblurDoGX shader",
			code: deblurDoGX_default
		});
		const deblurDoGXPipelineLayout = device.createPipelineLayout({
			label: "deblurDoGX pipeline layout",
			bindGroupLayouts: [deblurDoGXBindGroupLayout]
		});
		const deblurDoGXPipeline = device.createComputePipeline({
			label: "deblurDoGX pipeline",
			layout: deblurDoGXPipelineLayout,
			compute: {
				module: deblurDoGXModule,
				entryPoint: "computeMain"
			}
		});
		const deblurDoGXTexture = device.createTexture({
			size: [
				this.inputTexWidth,
				this.inputTexHeight,
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		const deblurDoGYBindGroupLayout = device.createBindGroupLayout({
			label: "deblurDoGY Bind Group Layout",
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.COMPUTE,
				texture: {}
			}, {
				binding: 1,
				visibility: GPUShaderStage.COMPUTE,
				storageTexture: {
					access: "write-only",
					format: "rgba16float"
				}
			}]
		});
		const deblurDoGYModule = device.createShaderModule({
			label: "deblurDoGY shader",
			code: deblurDoGY_default
		});
		const deblurDoGYPipelineLayout = device.createPipelineLayout({
			label: "deblurDoGY pipeline layout",
			bindGroupLayouts: [deblurDoGYBindGroupLayout]
		});
		const deblurDoGYPipeline = device.createComputePipeline({
			label: "deblurDoGY pipeline",
			layout: deblurDoGYPipelineLayout,
			compute: {
				module: deblurDoGYModule,
				entryPoint: "computeMain"
			}
		});
		const deblurDoGYTexture = device.createTexture({
			size: [
				this.inputTexWidth,
				this.inputTexHeight,
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		const deblurDoGApplyBindGroupLayout = device.createBindGroupLayout({
			label: "deblurDoGApply Bind Group Layout",
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					texture: {}
				},
				{
					binding: 1,
					visibility: GPUShaderStage.COMPUTE,
					texture: {}
				},
				{
					binding: 2,
					visibility: GPUShaderStage.COMPUTE,
					texture: {}
				},
				{
					binding: 3,
					visibility: GPUShaderStage.COMPUTE,
					storageTexture: {
						access: "write-only",
						format: "rgba16float"
					}
				},
				{
					binding: 4,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "uniform" }
				}
			]
		});
		const deblurDoGApplyModule = device.createShaderModule({
			label: "deblurDoGApply shader",
			code: deblurDoGApply_default
		});
		const deblurDoGApplyPipelineLayout = device.createPipelineLayout({
			label: "deblurDoGApply pipeline layout",
			bindGroupLayouts: [deblurDoGApplyBindGroupLayout]
		});
		const deblurDoGApplyPipeline = device.createComputePipeline({
			label: "deblurDoGApply pipeline",
			layout: deblurDoGApplyPipelineLayout,
			compute: {
				module: deblurDoGApplyModule,
				entryPoint: "computeMain"
			}
		});
		const deblurDoGApplyTexture = device.createTexture({
			size: [
				this.inputTexWidth,
				this.inputTexHeight,
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		this.textures = [
			luminationTexture,
			deblurDoGXTexture,
			deblurDoGYTexture,
			deblurDoGApplyTexture
		];
		this.modules = [
			luminationModule,
			deblurDoGXModule,
			deblurDoGYModule,
			deblurDoGApplyModule
		];
		this.bindGroupLayouts = [
			luminationBindGroupLayout,
			deblurDoGXBindGroupLayout,
			deblurDoGYBindGroupLayout,
			deblurDoGApplyBindGroupLayout
		];
		this.pipelineLayouts = [
			luminationPipelineLayout,
			deblurDoGXPipelineLayout,
			deblurDoGYPipelineLayout,
			deblurDoGApplyPipelineLayout
		];
		this.pipelines = [
			luminationPipeline,
			deblurDoGXPipeline,
			deblurDoGYPipeline,
			deblurDoGApplyPipeline
		];
		this.strengthBuffer = device.createBuffer({
			size: 4,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		const luminationBindGroup = this.device.createBindGroup({
			layout: this.bindGroupLayouts[0],
			entries: [{
				binding: 0,
				resource: this.inputTexture.createView()
			}, {
				binding: 1,
				resource: this.textures[0].createView()
			}]
		});
		const deblurDoGXBindGroup = this.device.createBindGroup({
			layout: this.bindGroupLayouts[1],
			entries: [{
				binding: 0,
				resource: this.textures[0].createView()
			}, {
				binding: 1,
				resource: this.textures[1].createView()
			}]
		});
		const deblurDoGYBindGroup = this.device.createBindGroup({
			layout: this.bindGroupLayouts[2],
			entries: [{
				binding: 0,
				resource: this.textures[1].createView()
			}, {
				binding: 1,
				resource: this.textures[2].createView()
			}]
		});
		const deblurDoGApplyBindGroup = this.device.createBindGroup({
			layout: this.bindGroupLayouts[3],
			entries: [
				{
					binding: 0,
					resource: this.textures[2].createView()
				},
				{
					binding: 1,
					resource: this.textures[0].createView()
				},
				{
					binding: 2,
					resource: this.inputTexture.createView()
				},
				{
					binding: 3,
					resource: this.textures[3].createView()
				},
				{
					binding: 4,
					resource: { buffer: this.strengthBuffer }
				}
			]
		});
		this.bindGroups = [
			luminationBindGroup,
			deblurDoGXBindGroup,
			deblurDoGYBindGroup,
			deblurDoGApplyBindGroup
		];
	}
	getOutputTexture() {
		return this.textures[3];
	}
	updateParam(param, value) {
		if (param !== "strength") throw new Error(`No param name as ${param}`);
		if (typeof value !== "number") throw new Error("strength must be a number");
		if (value < 0) throw new Error(`negative strength (${value}) is not allowed`);
		this.device.queue.writeBuffer(this.strengthBuffer, 0, new Float32Array([value]));
	}
	pass(encoder) {
		const luminationPass = encoder.beginComputePass();
		luminationPass.setPipeline(this.pipelines[0]);
		luminationPass.setBindGroup(0, this.bindGroups[0]);
		luminationPass.dispatchWorkgroups(Math.ceil(this.inputTexWidth / 8), Math.ceil(this.inputTexHeight / 8));
		luminationPass.end();
		const deblurDoGXPass = encoder.beginComputePass();
		deblurDoGXPass.setPipeline(this.pipelines[1]);
		deblurDoGXPass.setBindGroup(0, this.bindGroups[1]);
		deblurDoGXPass.dispatchWorkgroups(Math.ceil(this.inputTexWidth / 8), Math.ceil(this.inputTexHeight / 8));
		deblurDoGXPass.end();
		const deblurDoGYPass = encoder.beginComputePass();
		deblurDoGYPass.setPipeline(this.pipelines[2]);
		deblurDoGYPass.setBindGroup(0, this.bindGroups[2]);
		deblurDoGYPass.dispatchWorkgroups(Math.ceil(this.inputTexWidth / 8), Math.ceil(this.inputTexHeight / 8));
		deblurDoGYPass.end();
		const deblurDoGApplyPass = encoder.beginComputePass();
		deblurDoGApplyPass.setPipeline(this.pipelines[3]);
		deblurDoGApplyPass.setBindGroup(0, this.bindGroups[3]);
		deblurDoGApplyPass.dispatchWorkgroups(Math.ceil(this.inputTexWidth / 8), Math.ceil(this.inputTexHeight / 8));
		deblurDoGApplyPass.end();
	}
};
//#endregion
export { DoG };
