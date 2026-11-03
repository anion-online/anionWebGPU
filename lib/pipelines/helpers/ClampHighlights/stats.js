
import { launcher } from "../FrameGate/index.js";
import stats_default from "./shaders/stats.js";
import dering_default from "./shaders/dering.js";
//#region src/pipelines/helpers/ClampHighlights/stats.ts
/**
* Первая половина Clamp Highlights: максимум яркости 5×5 по исходнику
* (r32float). Вторую — зажим — выполняет последнее звено цепочки, если ему
* передан `deRing: stats.getOutputTexture()` (CNN-модели, Downscale): так
* не нужен отдельный проход в разрешении выхода.
*/
var ClampStats = class {
	name;
	pipeline;
	bindGroup;
	outputTexture;
	ready;
	launch;
	constructor({ device, inputTexture, gate, name = "clamp stats" }) {
		this.name = name;
		this.outputTexture = device.createTexture({
			label: `${name}: statsmax_texture`,
			size: [
				inputTexture.width,
				inputTexture.height,
				1
			],
			format: "r32float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		this.launch = launcher(gate, Math.ceil(inputTexture.width / 8), Math.ceil(inputTexture.height / 8));
		const layout = device.createBindGroupLayout({
			label: `${name} layout`,
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
		this.ready = device.createComputePipelineAsync({
			label: `${name} pipeline`,
			layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
			compute: {
				module: device.createShaderModule({
					label: `${name}: module`,
					code: stats_default
				}),
				entryPoint: "computeMain"
			}
		}).then((pipeline) => {
			this.pipeline = pipeline;
		});
		this.bindGroup = device.createBindGroup({
			label: `${name} bind group`,
			layout,
			entries: [{
				binding: 0,
				resource: inputTexture.createView()
			}, {
				binding: 1,
				resource: this.outputTexture.createView()
			}]
		});
	}
	updateParam(param, value) {
		throw new Error(`${this.name} has no param.`);
	}
	pass(encoder) {
		if (!this.pipeline) throw new Error(`${this.name}: шейдер ещё компилируется — дождитесь ready.`);
		const pass = encoder.beginComputePass({ label: this.name });
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		this.launch.dispatch(pass);
		pass.end();
	}
	getOutputTexture() {
		return this.outputTexture;
	}
};
/**
* Эпилог зажима для последнего звена: WGSL-функция deRing() и группа
* привязок 1. Без статистики группа 1 привязана к заглушке, а DERING = false.
*/
var DeRingEpilogue = class {
	static wgsl = dering_default;
	layout;
	bindGroup;
	constants;
	constructor(device, stats) {
		this.layout = device.createBindGroupLayout({
			label: "de-ring layout",
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.COMPUTE,
				texture: { sampleType: "unfilterable-float" }
			}]
		});
		const texture = stats ?? device.createTexture({
			label: "de-ring stub",
			size: [
				1,
				1,
				1
			],
			format: "r32float",
			usage: GPUTextureUsage.TEXTURE_BINDING
		});
		this.bindGroup = device.createBindGroup({
			label: "de-ring bind group",
			layout: this.layout,
			entries: [{
				binding: 0,
				resource: texture.createView()
			}]
		});
		this.constants = { DERING: stats ? 1 : 0 };
	}
};
//#endregion
export { ClampStats, DeRingEpilogue };
