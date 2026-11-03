
import stats_default from "./shaders/stats.js";
import { ClampStats, DeRingEpilogue } from "./stats.js";
import clamp_default from "./shaders/clamp.js";
//#region src/pipelines/helpers/ClampHighlights/index.ts
/**
* Anime4K Clamp Highlights (de-ring).
*
* Статистика (максимум яркости 5×5) снимается с исходника `statsTexture`, а
* зажим применяется к `inputTexture` — результату всей цепочки, в его
* разрешении. Поэтому звено ставится последним. В 1.0.0 оба шага шли по одному
* кадру в начале цепочки, и зажим ничего не делал: окно включает сам пиксель.
*/
var ClampHighlights = class {
	name;
	pipelines;
	bindGroups;
	statsTexture;
	outputTexture;
	constructor({ device, inputTexture, statsTexture, name = "clamp highlights" }) {
		this.name = name;
		this.statsTexture = device.createTexture({
			label: `${name}: statsmax_texture`,
			size: [
				statsTexture.width,
				statsTexture.height,
				1
			],
			format: "r32float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		this.outputTexture = device.createTexture({
			label: `${name}: clamp_highlights_texture`,
			size: [
				inputTexture.width,
				inputTexture.height,
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		const statsBindGroupLayout = device.createBindGroupLayout({
			label: `${name} stats bind group layout`,
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.COMPUTE,
				texture: {}
			}, {
				binding: 1,
				visibility: GPUShaderStage.COMPUTE,
				storageTexture: {
					access: "write-only",
					format: "r32float"
				}
			}]
		});
		const clampBindGroupLayout = device.createBindGroupLayout({
			label: `${name} clamp bind group layout`,
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					texture: {}
				},
				{
					binding: 1,
					visibility: GPUShaderStage.COMPUTE,
					texture: { sampleType: "unfilterable-float" }
				},
				{
					binding: 2,
					visibility: GPUShaderStage.COMPUTE,
					storageTexture: {
						access: "write-only",
						format: "rgba16float"
					}
				}
			]
		});
		const statsPipeline = device.createComputePipeline({
			label: `${name} stats pipeline`,
			layout: device.createPipelineLayout({ bindGroupLayouts: [statsBindGroupLayout] }),
			compute: {
				module: device.createShaderModule({
					label: `${name}: stats module`,
					code: stats_default
				}),
				entryPoint: "computeMain"
			}
		});
		const clampPipeline = device.createComputePipeline({
			label: `${name} clamp pipeline`,
			layout: device.createPipelineLayout({ bindGroupLayouts: [clampBindGroupLayout] }),
			compute: {
				module: device.createShaderModule({
					label: `${name}: clamp module`,
					code: clamp_default
				}),
				entryPoint: "computeMain"
			}
		});
		this.pipelines = {
			statsPipeline,
			clampPipeline
		};
		this.bindGroups = {
			statsBindGroup: device.createBindGroup({
				label: `${name} stats bind group`,
				layout: statsBindGroupLayout,
				entries: [{
					binding: 0,
					resource: statsTexture.createView()
				}, {
					binding: 1,
					resource: this.statsTexture.createView()
				}]
			}),
			clampBindGroup: device.createBindGroup({
				label: `${name} clamp bind group`,
				layout: clampBindGroupLayout,
				entries: [
					{
						binding: 0,
						resource: inputTexture.createView()
					},
					{
						binding: 1,
						resource: this.statsTexture.createView()
					},
					{
						binding: 2,
						resource: this.outputTexture.createView()
					}
				]
			})
		};
	}
	updateParam(param, value) {
		throw new Error(`${this.name} has no param.`);
	}
	pass(encoder) {
		const statsPass = encoder.beginComputePass();
		statsPass.setPipeline(this.pipelines.statsPipeline);
		statsPass.setBindGroup(0, this.bindGroups.statsBindGroup);
		statsPass.dispatchWorkgroups(Math.ceil(this.statsTexture.width / 8), Math.ceil(this.statsTexture.height / 8));
		statsPass.end();
		const clampPass = encoder.beginComputePass();
		clampPass.setPipeline(this.pipelines.clampPipeline);
		clampPass.setBindGroup(0, this.bindGroups.clampBindGroup);
		clampPass.dispatchWorkgroups(Math.ceil(this.outputTexture.width / 8), Math.ceil(this.outputTexture.height / 8));
		clampPass.end();
	}
	getOutputTexture() {
		return this.outputTexture;
	}
};
//#endregion
export { ClampHighlights, ClampStats, DeRingEpilogue };
