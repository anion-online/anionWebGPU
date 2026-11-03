
import { launcher } from "../../helpers/FrameGate/index.js";
import deband_default from "./shaders/deband.js";
//#region src/pipelines/denoise/Deband/index.ts
const DEBAND_DEFAULTS = {
	threshold: 2,
	range: 16,
	iterations: 2
};
/**
* Дебандинг до сети (Э21): Kodik после x264 приносит полосы на градиентах —
* небо, дым, тёмные сцены. Сеть ступеньку принимает за край и прорисовывает,
* дизеринг финального blit от них не спасает — он лишь не добавляет своих.
* Выход в rgba16float: сглаженный градиент сохраняет точность выше 8 бит.
*/
var Deband = class {
	name;
	ready;
	device;
	pipeline;
	bindGroup;
	params;
	values;
	launch;
	outputTexture;
	constructor({ device, inputTexture, gate, name = "deband", ...params }) {
		this.name = name;
		this.device = device;
		const { width, height } = inputTexture;
		this.outputTexture = device.createTexture({
			label: `${name} output texture`,
			size: [
				width,
				height,
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		this.params = device.createBuffer({
			label: `${name} params`,
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.values = { ...DEBAND_DEFAULTS };
		this.set(params);
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
				},
				{
					binding: 3,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "uniform" }
				}
			]
		});
		this.bindGroup = device.createBindGroup({
			label: `${name} bind group`,
			layout,
			entries: [
				{
					binding: 0,
					resource: inputTexture.createView()
				},
				{
					binding: 1,
					resource: device.createSampler({
						magFilter: "linear",
						minFilter: "linear"
					})
				},
				{
					binding: 2,
					resource: this.outputTexture.createView()
				},
				{
					binding: 3,
					resource: { buffer: this.params }
				}
			]
		});
		this.launch = launcher(gate, Math.ceil(width / 8), Math.ceil(height / 8));
		this.ready = device.createComputePipelineAsync({
			label: `${name} pipeline`,
			layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
			compute: {
				module: device.createShaderModule({
					label: `${name} module`,
					code: deband_default
				}),
				entryPoint: "computeMain"
			}
		}).then((pipeline) => {
			this.pipeline = pipeline;
		});
	}
	/** Новые параметры; неуказанные остаются прежними. */
	set(params) {
		const defined = Object.fromEntries(Object.entries(params).filter(([, v]) => v !== void 0));
		this.values = {
			...this.values,
			...defined
		};
		const { threshold, range, iterations } = this.values;
		const data = /* @__PURE__ */ new ArrayBuffer(16);
		new Float32Array(data, 0, 2).set([Math.max(0, threshold) / 255, Math.max(0, range)]);
		new Uint32Array(data, 8, 1).set([Math.max(0, Math.floor(iterations))]);
		this.device.queue.writeBuffer(this.params, 0, data);
	}
	updateParam(param, value) {
		if (param !== "threshold" && param !== "range" && param !== "iterations") throw new Error(`${this.name} has no param ${param}.`);
		this.set({ [param]: Number(value) });
	}
	pass(encoder) {
		if (!this.pipeline) throw new Error(`${this.name}: шейдеры ещё компилируются — дождитесь ready.`);
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
//#endregion
export { DEBAND_DEFAULTS, Deband };
