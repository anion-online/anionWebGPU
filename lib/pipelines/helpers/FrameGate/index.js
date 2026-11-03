
import diff_default from "./shaders/diff.js";
import decide_default from "./shaders/decide.js";
import copy_default from "./shaders/copy.js";
//#region src/pipelines/helpers/FrameGate/index.ts
/** Слотов хватает на любую цепочку ModeC/ModeCA (их там до ~25). */
const CAPACITY = 64;
const SLOT_BYTES = 12;
const storage = (type) => ({ type });
/**
* Запуск звена с `gate` — косвенный из буфера ворот, без него — как раньше.
* Число групп известно при сборке звена, поэтому слот берётся там же.
*/
function launcher(gate, x, y) {
	if (gate) return gate.slot(x, y);
	return { dispatch: (pass) => pass.dispatchWorkgroups(x, y) };
}
/**
* Ворота повторов. В аниме много кадров подряд с тем же рисунком (анимация
* «на двойках», статичные планы), а цепочка CNN каждый раз считала их заново.
* Ворота сравнивают новый кадр с последним посчитанным и на повторе обнуляют
* аргументы косвенного запуска всех звеньев: GPU пропускает работу, выход
* остаётся прошлым. Решение принимается на GPU — CPU не ждёт чтения.
*
* Звено ставится первым в цепочке; остальные получают `gate` и запускаются
* через `launcher()`.
*/
var FrameGate = class {
	name;
	ready;
	/** Аргументы косвенного запуска звеньев (INDIRECT). */
	args;
	device;
	template;
	state;
	readback;
	reading;
	prev;
	input;
	slots = 0;
	width;
	height;
	pipelines = {};
	bindGroups;
	copyLaunch;
	constructor({ device, inputTexture, threshold = 0, meanThreshold = .5, name = "frame gate" }) {
		this.name = name;
		this.device = device;
		this.input = inputTexture;
		this.width = inputTexture.width;
		this.height = inputTexture.height;
		this.args = device.createBuffer({
			label: `${name}: args`,
			size: 768,
			usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE
		});
		this.template = device.createBuffer({
			label: `${name}: template`,
			size: 768,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		this.state = device.createBuffer({
			label: `${name}: state`,
			size: 64,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
		});
		this.prev = device.createTexture({
			label: `${name}: previous frame`,
			size: [
				this.width,
				this.height,
				1
			],
			format: "rgba8unorm",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		this.setThreshold(threshold);
		this.setMeanThreshold(meanThreshold);
		this.invalidate();
		this.copyLaunch = this.slot(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
		const diffLayout = device.createBindGroupLayout({
			label: `${name}: diff layout`,
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
					buffer: storage("storage")
				}
			]
		});
		const decideLayout = device.createBindGroupLayout({
			label: `${name}: decide layout`,
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					buffer: storage("storage")
				},
				{
					binding: 1,
					visibility: GPUShaderStage.COMPUTE,
					buffer: storage("read-only-storage")
				},
				{
					binding: 2,
					visibility: GPUShaderStage.COMPUTE,
					buffer: storage("storage")
				}
			]
		});
		const copyLayout = device.createBindGroupLayout({
			label: `${name}: copy layout`,
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.COMPUTE,
				texture: {}
			}, {
				binding: 1,
				visibility: GPUShaderStage.COMPUTE,
				storageTexture: {
					access: "write-only",
					format: "rgba8unorm"
				}
			}]
		});
		this.bindGroups = {
			diff: device.createBindGroup({
				label: `${name}: diff`,
				layout: diffLayout,
				entries: [
					{
						binding: 0,
						resource: inputTexture.createView()
					},
					{
						binding: 1,
						resource: this.prev.createView()
					},
					{
						binding: 2,
						resource: { buffer: this.state }
					}
				]
			}),
			decide: device.createBindGroup({
				label: `${name}: decide`,
				layout: decideLayout,
				entries: [
					{
						binding: 0,
						resource: { buffer: this.state }
					},
					{
						binding: 1,
						resource: { buffer: this.template }
					},
					{
						binding: 2,
						resource: { buffer: this.args }
					}
				]
			}),
			copy: device.createBindGroup({
				label: `${name}: copy`,
				layout: copyLayout,
				entries: [{
					binding: 0,
					resource: inputTexture.createView()
				}, {
					binding: 1,
					resource: this.prev.createView()
				}]
			})
		};
		const compile = (key, code, layout) => device.createComputePipelineAsync({
			label: `${name}: ${key}`,
			layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
			compute: {
				module: device.createShaderModule({
					label: `${name}: ${key}`,
					code
				}),
				entryPoint: "computeMain"
			}
		}).then((pipeline) => {
			this.pipelines[key] = pipeline;
		});
		this.ready = Promise.all([
			compile("diff", diff_default, diffLayout),
			compile("decide", decide_default, decideLayout),
			compile("copy", copy_default, copyLayout)
		]).then(() => void 0);
	}
	/** Слот косвенного запуска на `x × y` групп; звено зовёт при сборке. */
	slot(x, y) {
		if (this.slots >= CAPACITY) throw new Error(`${this.name}: больше ${CAPACITY} запусков в цепочке.`);
		const offset = this.slots * SLOT_BYTES;
		this.slots += 1;
		this.device.queue.writeBuffer(this.template, offset, new Uint32Array([
			x,
			y,
			1
		]));
		this.device.queue.writeBuffer(this.state, 16, new Uint32Array([this.slots]));
		return { dispatch: (pass) => pass.dispatchWorkgroupsIndirect(this.args, offset) };
	}
	/** Порог в уровнях 8 бит (см. `threshold`). */
	setThreshold(threshold) {
		this.device.queue.writeBuffer(this.state, 12, new Uint32Array([Math.max(0, Math.floor(threshold))]));
	}
	/** Порог среднего сдвига в уровнях 8 бит (см. `meanThreshold`). */
	setMeanThreshold(meanThreshold) {
		const limit = Math.floor(Math.max(0, meanThreshold) * this.width * this.height);
		this.device.queue.writeBuffer(this.state, 44, new Uint32Array([Math.min(limit, 2147483647)]));
	}
	/** Посчитать следующий кадр, даже если он повтор. */
	invalidate() {
		this.device.queue.writeBuffer(this.state, 20, new Uint32Array([1]));
	}
	/** Счётчики с GPU; параллельные вызовы получают одно чтение. */
	readStats() {
		if (this.reading) return this.reading;
		this.readback ??= this.device.createBuffer({
			label: `${this.name}: readback`,
			size: 32,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		});
		const readback = this.readback;
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.state, 0, readback, 0, 32);
		this.device.queue.submit([encoder.finish()]);
		this.reading = readback.mapAsync(GPUMapMode.READ).then(() => {
			const s = new Uint32Array(readback.getMappedRange().slice(0));
			readback.unmap();
			return {
				frames: s[1],
				skipped: s[2],
				lastDiff: s[6]
			};
		}).finally(() => {
			this.reading = void 0;
		});
		return this.reading;
	}
	updateParam(param, value) {
		if (param === "threshold") {
			this.setThreshold(Number(value));
			return;
		}
		if (param === "meanThreshold") {
			this.setMeanThreshold(Number(value));
			return;
		}
		throw new Error(`${this.name} has no param ${param}.`);
	}
	pass(encoder) {
		const { diff, decide, copy } = this.pipelines;
		if (!diff || !decide || !copy) throw new Error(`${this.name}: шейдеры ещё компилируются — дождитесь ready.`);
		encoder.clearBuffer(this.state, 0, 4);
		encoder.clearBuffer(this.state, 32, 12);
		const pass = encoder.beginComputePass({ label: this.name });
		pass.setPipeline(diff);
		pass.setBindGroup(0, this.bindGroups.diff);
		pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
		pass.setPipeline(decide);
		pass.setBindGroup(0, this.bindGroups.decide);
		pass.dispatchWorkgroups(1);
		pass.setPipeline(copy);
		pass.setBindGroup(0, this.bindGroups.copy);
		this.copyLaunch.dispatch(pass);
		pass.end();
	}
	/** Ворота ничего не выводят — отдают вход как есть. */
	getOutputTexture() {
		return this.input;
	}
};
//#endregion
export { FrameGate, launcher };
