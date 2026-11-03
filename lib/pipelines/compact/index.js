
import { launcher } from "../helpers/FrameGate/index.js";
import { WINO_REGION, chromaShader, groups, kernelRegion, layerBases, layerMatrices, layerShader, packMatrices, winoLayerShader, winoMatrices } from "./shaders.js";
//#region src/pipelines/compact/index.ts
/**
* SRVGGNetCompact ×2 (AnimeJaNai V2 SuperUltraCompact, режим «Детали»):
* слой — compute-проход, признаки между слоями — планарные буферы f16
* (6 × vec4<f16> на пиксель), выход — текстура rgba16float ×2. С
* `chromaSigma` перед слоями — два прохода сглаживания цветности кадра, а
* последний слой берёт от сети только яркость. `SRVGGNetCompactU` — слои
* на сетке вдвое реже кадра (вход pixel-unshuffle 2×2, выход pixelshuffle ×4).
*/
var CompactSR = class {
	name;
	ready;
	outputTexture;
	steps = [];
	launch;
	constructor({ device, inputTexture, model, kernel = "fast", gate, chromaSigma = 0, chromaThreshold = 0, name = "CompactSR" }) {
		this.name = name;
		const { meta, weights } = model;
		const unshuffle = meta.arch === "SRVGGNetCompactU";
		if (!unshuffle && meta.arch !== "SRVGGNetCompact") throw new Error(`${name}: архитектура ${meta.arch} не поддерживается`);
		if (unshuffle && meta.unshuffle !== 2) throw new Error(`${name}: unshuffle ${meta.unshuffle} не поддерживается`);
		if (weights.byteLength !== meta.bytes) throw new Error(`${name}: веса ${weights.byteLength} Б, в описании ${meta.bytes} Б`);
		if (!device.features.has("shader-f16")) throw new Error(`${name}: нужна фича 'shader-f16'`);
		const { width, height } = inputTexture;
		const { scale } = meta;
		if (unshuffle && (kernel === "reference" || kernel === "wino")) throw new Error(`${name}: у половинной сетки нет ядра ${kernel}`);
		const gridW = unshuffle ? Math.ceil(width / 2) : width;
		const gridH = unshuffle ? Math.ceil(height / 2) : height;
		const wino = kernel === "wino";
		const direct = wino ? "fast" : kernel;
		const [regionX, regionY] = kernelRegion(direct);
		this.launch = launcher(gate, Math.ceil(gridW / regionX), Math.ceil(gridH / regionY));
		const winoLaunch = wino ? launcher(gate, Math.ceil(width / WINO_REGION[0]), Math.ceil(height / WINO_REGION[1])) : void 0;
		this.outputTexture = device.createTexture({
			label: `${name}: output`,
			size: [
				width * scale,
				height * scale,
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
		const featBytes = gridW * gridH * groups(meta.numFeat) * 8;
		const ping = device.createBuffer({
			label: `${name}: ping`,
			size: featBytes,
			usage: storage
		});
		const pong = device.createBuffer({
			label: `${name}: pong`,
			size: featBytes,
			usage: storage
		});
		const fast = kernel !== "reference";
		const raw = new Uint16Array(weights);
		const packed = fast ? packMatrices(meta) : null;
		const halves = packed ? Uint16Array.from(packed.order, (i) => i === 4294967295 ? 0 : raw[i]) : raw;
		const blob = device.createBuffer({
			label: `${name}: weights`,
			size: halves.byteLength,
			usage: storage
		});
		device.queue.writeBuffer(blob, 0, halves);
		const compiling = [];
		const chroma = chromaSigma > 0;
		let chromaView;
		let chromaSampler;
		if (chroma) {
			const chromaTexture = (label) => device.createTexture({
				label: `${name}: ${label}`,
				size: [
					width,
					height,
					1
				],
				format: "rgba16float",
				usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
			});
			const across = chromaTexture("chroma x");
			const blurred = chromaTexture("chroma");
			chromaView = blurred.createView();
			chromaSampler = device.createSampler({
				magFilter: "linear",
				minFilter: "linear"
			});
			const chromaLaunch = launcher(gate, Math.ceil(width / 8), Math.ceil(height / 8));
			[[
				"x",
				inputTexture,
				across
			], [
				"y",
				across,
				blurred
			]].forEach(([axis, from, to]) => {
				const step = {
					bindGroup: void 0,
					launch: chromaLaunch
				};
				this.steps.push(step);
				compiling.push(device.createComputePipelineAsync({
					label: `${name}: chroma ${axis}`,
					layout: "auto",
					compute: {
						module: device.createShaderModule({
							label: `${name}: chroma ${axis}`,
							code: chromaShader(axis, chromaSigma, chromaThreshold)
						}),
						entryPoint: "main"
					}
				}).then((pipeline) => {
					step.pipeline = pipeline;
					step.bindGroup = device.createBindGroup({
						label: `${name}: chroma ${axis}`,
						layout: pipeline.getBindGroupLayout(0),
						entries: [{
							binding: 0,
							resource: from.createView()
						}, {
							binding: 1,
							resource: to.createView()
						}]
					});
				}));
			});
		}
		let src = null;
		let dst = ping;
		meta.layers.forEach((layer, index) => {
			const first = index === 0;
			const last = index === meta.layers.length - 1;
			const inner = wino && !first && !last;
			const bases = packed ? packed.bases[index] : layerBases(layer);
			const dims = device.createBuffer({
				label: `${name}: dims ${index}`,
				size: 32,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
			device.queue.writeBuffer(dims, 0, new Uint32Array([
				gridW,
				gridH,
				bases.w,
				bases.bias,
				bases.prelu,
				0,
				0,
				0
			]));
			const entries = [
				{
					binding: 0,
					resource: { buffer: dims }
				},
				{
					binding: 1,
					resource: first ? inputTexture.createView() : { buffer: src }
				},
				{
					binding: 2,
					resource: last ? this.outputTexture.createView() : { buffer: dst }
				}
			];
			entries.push({
				binding: 3,
				resource: { buffer: blob }
			});
			if (last) entries.push({
				binding: 4,
				resource: inputTexture.createView()
			});
			if (last && chromaView && chromaSampler) entries.push({
				binding: 6,
				resource: chromaSampler
			}, {
				binding: 7,
				resource: chromaView
			});
			if (inner) {
				const transformed = winoMatrices(layer, raw);
				const matrices = device.createBuffer({
					label: `${name}: weights ${index}`,
					size: transformed.byteLength,
					usage: storage
				});
				device.queue.writeBuffer(matrices, 0, transformed);
				entries.push({
					binding: 5,
					resource: { buffer: matrices }
				});
			} else if (packed) {
				const size = layerMatrices(layer) * 32;
				const matrices = device.createBuffer({
					label: `${name}: weights ${index}`,
					size,
					usage: storage
				});
				device.queue.writeBuffer(matrices, 0, halves, bases.w * 4, size / 2);
				entries.push({
					binding: 5,
					resource: { buffer: matrices }
				});
			}
			const module = device.createShaderModule({
				label: `${name}: layer ${index}`,
				code: inner ? winoLayerShader(layer) : layerShader({
					kernel: direct,
					layer,
					first,
					tailScale: last ? scale : void 0,
					chroma,
					unshuffle
				})
			});
			const step = {
				bindGroup: void 0,
				launch: inner ? winoLaunch : void 0
			};
			this.steps.push(step);
			compiling.push(device.createComputePipelineAsync({
				label: `${name}: layer ${index}`,
				layout: "auto",
				compute: {
					module,
					entryPoint: "main"
				}
			}).then((pipeline) => {
				step.pipeline = pipeline;
				step.bindGroup = device.createBindGroup({
					label: `${name}: layer ${index}`,
					layout: pipeline.getBindGroupLayout(0),
					entries
				});
			}));
			if (!last) {
				src = dst;
				dst = dst === ping ? pong : ping;
			}
		});
		this.ready = Promise.all(compiling).then(() => void 0);
	}
	updateParam(param, value) {
		throw new Error(`${this.name} has no param.`);
	}
	pass(encoder) {
		const pass = encoder.beginComputePass({ label: this.name });
		this.steps.forEach((step) => {
			if (!step.pipeline) throw new Error(`${this.name}: шейдеры ещё компилируются — дождитесь ready.`);
			pass.setPipeline(step.pipeline);
			pass.setBindGroup(0, step.bindGroup);
			(step.launch ?? this.launch).dispatch(pass);
		});
		pass.end();
	}
	getOutputTexture() {
		return this.outputTexture;
	}
};
//#endregion
export { CompactSR };
