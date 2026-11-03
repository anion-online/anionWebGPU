
import stage0_default from "./shaders/stage0.js";
import stage1_default from "./shaders/stage1.js";
import stage2_default from "./shaders/stage2.js";
import stage3_default from "./shaders/stage3.js";
//#region src/pipelines/upscale/DenoiseCNNx2L/model.ts
const model = {
	scale: 2,
	block: [1, 1],
	textures: 4,
	packed: [
		1,
		2,
		3
	],
	stages: [
		{
			wgsl: stage0_default,
			inputs: [0],
			outputs: [1]
		},
		{
			wgsl: stage1_default,
			inputs: [1],
			outputs: [2]
		},
		{
			wgsl: stage2_default,
			inputs: [2],
			outputs: [3]
		},
		{
			wgsl: stage3_default,
			inputs: [3],
			outputs: [],
			final: true
		}
	]
};
//#endregion
export { model as default };
