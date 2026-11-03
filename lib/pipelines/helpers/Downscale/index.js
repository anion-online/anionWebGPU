
import { launcher } from "../FrameGate/index.js";
import downscale_default from "./shaders/downscale.js";
import { DeRingEpilogue } from "../ClampHighlights/stats.js";
//#region src/pipelines/helpers/Downscale/index.ts
/**
* Уменьшение до `targetDimensions`. По умолчанию — Catmull-Rom с ядром,
* растянутым на коэффициент уменьшения (без алиасинга); 'bilinear' — выборка
* в точке, как AutoDownscalePre в mpv и 1.0.0. С `deRing` последний проход
* сразу выполняет Clamp Highlights.
*/
var Downscale = class {
	outputTexture;
	steps;
	ready;
	/** Группа 1: статистика для deRing() или заглушка. */
	deRing;
	name;
	constructor({ device, inputTexture, targetDimensions, filter = "catmull-rom", deRing, gate, name = "downscale" }) {
		this.name = name;
		this.outputTexture = device.createTexture({
			label: `${name} output texture`,
			size: [
				targetDimensions.width,
				targetDimensions.height,
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		const module = device.createShaderModule({
			label: `${name} module`,
			code: DeRingEpilogue.wgsl + downscale_default
		});
		const sampler = device.createSampler({
			magFilter: "linear",
			minFilter: "linear"
		});
		const epilogue = new DeRingEpilogue(device, deRing);
		const layout = device.createBindGroupLayout({
			label: `${name} layout`,
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					texture: {}
				},
				{
					binding: 1,
					visibility: GPUShaderStage.COMPUTE,
					sampler: {}
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
		const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout, epilogue.layout] });
		const compiling = [];
		const step = (constants, input, output, last) => {
			const built = {
				bindGroup: device.createBindGroup({
					label: `${name} bind group`,
					layout,
					entries: [
						{
							binding: 0,
							resource: input.createView()
						},
						{
							binding: 1,
							resource: sampler
						},
						{
							binding: 2,
							resource: output.createView()
						}
					]
				}),
				output,
				launch: launcher(gate, Math.ceil(output.width / 8), Math.ceil(output.height / 8))
			};
			compiling.push(device.createComputePipelineAsync({
				label: `${name} pipeline`,
				layout: pipelineLayout,
				compute: {
					module,
					entryPoint: "computeMain",
					constants: {
						...constants,
						DERING: last ? epilogue.constants.DERING : 0
					}
				}
			}).then((pipeline) => {
				built.pipeline = pipeline;
			}));
			return built;
		};
		this.deRing = epilogue.bindGroup;
		if (filter === "bilinear") this.steps = [step({ FILTER: 0 }, inputTexture, this.outputTexture, true)];
		else {
			const middle = device.createTexture({
				label: `${name} middle texture`,
				size: [
					targetDimensions.width,
					inputTexture.height,
					1
				],
				format: "rgba16float",
				usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
			});
			this.steps = [step({
				FILTER: 1,
				AXIS: 0
			}, inputTexture, middle, false), step({
				FILTER: 1,
				AXIS: 1
			}, middle, this.outputTexture, true)];
		}
		this.ready = Promise.all(compiling).then(() => void 0);
	}
	updateParam(param, value) {
		throw new Error(`${this.name} has no param`);
	}
	pass(encoder) {
		const pass = encoder.beginComputePass({ label: this.name });
		this.steps.forEach(({ pipeline, bindGroup, launch }) => {
			if (!pipeline) throw new Error(`${this.name}: шейдеры ещё компилируются — дождитесь ready.`);
			pass.setPipeline(pipeline);
			pass.setBindGroup(0, bindGroup);
			pass.setBindGroup(1, this.deRing);
			launch.dispatch(pass);
		});
		pass.end();
	}
	getOutputTexture() {
		return this.outputTexture;
	}
};
//#endregion
export { Downscale };
