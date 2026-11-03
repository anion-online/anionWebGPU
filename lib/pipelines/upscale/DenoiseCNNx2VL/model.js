
import stage0_default from "./shaders/stage0.js";
import stage1_default from "./shaders/stage1.js";
import stage2_default from "./shaders/stage2.js";
import stage3_default from "./shaders/stage3.js";
import stage4_default from "./shaders/stage4.js";
import stage5_default from "./shaders/stage5.js";
import stage6_default from "./shaders/stage6.js";
import stage7_default from "./shaders/stage7.js";
//#region src/pipelines/upscale/DenoiseCNNx2VL/model.ts
const model = {
	scale: 2,
	block: [1, 1],
	textures: 8,
	packed: [
		1,
		2,
		3,
		4,
		5,
		6,
		7
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
			outputs: [4]
		},
		{
			wgsl: stage4_default,
			inputs: [4],
			outputs: [5]
		},
		{
			wgsl: stage5_default,
			inputs: [5],
			outputs: [6]
		},
		{
			wgsl: stage6_default,
			inputs: [6],
			outputs: [7]
		},
		{
			wgsl: stage7_default,
			inputs: [
				1,
				2,
				3,
				4,
				5,
				6,
				7
			],
			outputs: [],
			final: true
		}
	]
};
//#endregion
export { model as default };
