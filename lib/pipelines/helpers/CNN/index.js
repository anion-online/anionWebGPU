
import { launcher } from "../FrameGate/index.js";
import { DeRingEpilogue } from "../ClampHighlights/stats.js";
//#region src/pipelines/helpers/CNN/index.ts
const PRELUDE = {
	f32: "alias T4 = vec4f;\nalias M4 = mat4x4f;\nalias A4 = vec4f;\nalias S1 = f32;\n",
	f16: "enable f16;\nalias T4 = vec4h;\nalias M4 = mat4x4h;\nalias A4 = vec4h;\nalias S1 = f16;\n"
};
const UNPACK = {
	f32: "fn unpack_half(a: u32, b: u32) -> T4 { return T4(unpack2x16float(a), unpack2x16float(b)); }\n",
	f16: "fn unpack_half(a: u32, b: u32) -> T4 { return T4(bitcast<vec2h>(a), bitcast<vec2h>(b)); }\n"
};
/**
* Исполнитель CNN-модели из conversion/cnn.py: стадия — один compute-проход,
* поток — один пиксель входа (финальная стадия x2 пишет 2×2 пикселя выхода).
*/
var CNN = class {
	name;
	precision;
	stages = [];
	ready;
	outputTexture;
	block;
	width;
	height;
	constructor({ device, inputTexture, model, name = "cnn", precision = "f32", deRing, gate }) {
		this.name = name;
		if (precision === "f16" && !device.features.has("shader-f16")) throw new Error(`${name}: precision 'f16' requires a device with 'shader-f16'.`);
		this.precision = precision;
		this.block = model.block;
		this.width = inputTexture.width;
		this.height = inputTexture.height;
		const packed = new Set(model.packed);
		const textures = [inputTexture];
		for (let i = 1; i < model.textures; i += 1) textures.push(device.createTexture({
			label: `${name}: feature ${i}`,
			size: [
				this.width,
				this.height,
				1
			],
			format: packed.has(i) ? "rgba32uint" : "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		}));
		this.outputTexture = device.createTexture({
			label: `${name}: output`,
			size: [
				this.width * model.scale,
				this.height * model.scale,
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		const sampler = device.createSampler({
			magFilter: "linear",
			minFilter: "linear"
		});
		const epilogue = new DeRingEpilogue(device, deRing);
		const compiling = [];
		model.stages.forEach((stage, n) => {
			const layoutEntries = [];
			const entries = [];
			const addTexture = (texture) => {
				const binding = layoutEntries.length;
				layoutEntries.push({
					binding,
					visibility: GPUShaderStage.COMPUTE,
					texture: texture.format === "rgba32uint" ? { sampleType: "uint" } : {}
				});
				entries.push({
					binding,
					resource: texture.createView()
				});
			};
			const addStorage = (texture) => {
				const binding = layoutEntries.length;
				layoutEntries.push({
					binding,
					visibility: GPUShaderStage.COMPUTE,
					storageTexture: {
						access: "write-only",
						format: texture.format
					}
				});
				entries.push({
					binding,
					resource: texture.createView()
				});
			};
			stage.inputs.forEach((i) => addTexture(textures[i]));
			if (stage.final) {
				addTexture(inputTexture);
				const binding = layoutEntries.length;
				layoutEntries.push({
					binding,
					visibility: GPUShaderStage.COMPUTE,
					sampler: {}
				});
				entries.push({
					binding,
					resource: sampler
				});
				addStorage(this.outputTexture);
			} else stage.outputs.forEach((i) => addStorage(textures[i]));
			const layout = device.createBindGroupLayout({
				label: `${name}: stage ${n} layout`,
				entries: layoutEntries
			});
			const final = Boolean(stage.final);
			const compiled = {
				bindGroup: device.createBindGroup({
					label: `${name}: stage ${n}`,
					layout,
					entries
				}),
				deRing: final ? epilogue.bindGroup : void 0,
				launch: launcher(gate, Math.ceil(this.width / (8 * this.block[0])), Math.ceil(this.height / (8 * this.block[1])))
			};
			this.stages.push(compiled);
			compiling.push(device.createComputePipelineAsync({
				label: `${name}: stage ${n}`,
				layout: device.createPipelineLayout({ bindGroupLayouts: final ? [layout, epilogue.layout] : [layout] }),
				compute: {
					module: device.createShaderModule({
						label: `${name}: stage ${n}`,
						code: (stage.subgroups ? "enable subgroups;\n" : "") + PRELUDE[this.precision] + (packed.size ? UNPACK[this.precision] : "") + (final ? DeRingEpilogue.wgsl : "") + stage.wgsl
					}),
					entryPoint: "computeMain",
					constants: final ? epilogue.constants : {}
				}
			}).then((pipeline) => {
				compiled.pipeline = pipeline;
			}));
		});
		this.ready = Promise.all(compiling).then(() => void 0);
	}
	updateParam(param, value) {
		throw new Error(`${this.name} has no param.`);
	}
	pass(encoder) {
		const pass = encoder.beginComputePass({ label: this.name });
		this.stages.forEach((stage) => {
			if (!stage.pipeline) throw new Error(`${this.name}: шейдеры ещё компилируются — дождитесь ready.`);
			pass.setPipeline(stage.pipeline);
			pass.setBindGroup(0, stage.bindGroup);
			if (stage.deRing) pass.setBindGroup(1, stage.deRing);
			stage.launch.dispatch(pass);
		});
		pass.end();
	}
	getOutputTexture() {
		return this.outputTexture;
	}
};
//#endregion
export { CNN };
