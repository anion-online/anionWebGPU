
import stage0_default from "./shaders/stage0.js";
import stage1_default from "./shaders/stage1.js";
import stage2_default from "./shaders/stage2.js";
import stage3_default from "./shaders/stage3.js";
import stage4_default from "./shaders/stage4.js";
import stage5_default from "./shaders/stage5.js";
import stage6_default from "./shaders/stage6.js";
//#region src/pipelines/artcnn/ArtCNNC4F16DS/model.ts
const model = {
	scale: 2,
	block: [1, 1],
	textures: 25,
	stages: [
		{
			wgsl: stage0_default,
			inputs: [0],
			outputs: [
				1,
				2,
				3,
				4
			]
		},
		{
			wgsl: stage1_default,
			inputs: [
				1,
				2,
				3,
				4
			],
			outputs: [
				5,
				6,
				7,
				8
			]
		},
		{
			wgsl: stage2_default,
			inputs: [
				5,
				6,
				7,
				8
			],
			outputs: [
				9,
				10,
				11,
				12
			]
		},
		{
			wgsl: stage3_default,
			inputs: [
				9,
				10,
				11,
				12
			],
			outputs: [
				13,
				14,
				15,
				16
			]
		},
		{
			wgsl: stage4_default,
			inputs: [
				13,
				14,
				15,
				16
			],
			outputs: [
				17,
				18,
				19,
				20
			]
		},
		{
			wgsl: stage5_default,
			inputs: [
				17,
				18,
				19,
				20
			],
			outputs: [
				21,
				22,
				23,
				24
			]
		},
		{
			wgsl: stage6_default,
			inputs: [
				21,
				1,
				22,
				2,
				23,
				3,
				24,
				4
			],
			outputs: [],
			final: true
		}
	]
};
//#endregion
export { model as default };
