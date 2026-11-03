
import stage0_default from "./shaders/stage0.js";
import stage1_default from "./shaders/stage1.js";
import stage2_default from "./shaders/stage2.js";
import stage3_default from "./shaders/stage3.js";
import stage4_default from "./shaders/stage4.js";
import stage5_default from "./shaders/stage5.js";
import stage6_default from "./shaders/stage6.js";
import stage7_default from "./shaders/stage7.js";
import stage8_default from "./shaders/stage8.js";
import stage9_default from "./shaders/stage9.js";
import stage10_default from "./shaders/stage10.js";
import stage11_default from "./shaders/stage11.js";
import stage12_default from "./shaders/stage12.js";
import stage13_default from "./shaders/stage13.js";
//#region src/pipelines/artcnn/ArtCNNC4F32DS/model.ts
const model = {
	scale: 2,
	block: [1, 1],
	textures: 50,
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
			inputs: [0],
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
				1,
				2,
				3,
				4,
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
				1,
				2,
				3,
				4,
				5,
				6,
				7,
				8
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
				9,
				10,
				11,
				12,
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
				9,
				10,
				11,
				12,
				13,
				14,
				15,
				16
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
				17,
				18,
				19,
				20,
				21,
				22,
				23,
				24
			],
			outputs: [
				25,
				26,
				27,
				28
			]
		},
		{
			wgsl: stage7_default,
			inputs: [
				17,
				18,
				19,
				20,
				21,
				22,
				23,
				24
			],
			outputs: [
				29,
				30,
				31,
				32
			]
		},
		{
			wgsl: stage8_default,
			inputs: [
				25,
				26,
				27,
				28,
				29,
				30,
				31,
				32
			],
			outputs: [
				33,
				34,
				35,
				36
			]
		},
		{
			wgsl: stage9_default,
			inputs: [
				25,
				26,
				27,
				28,
				29,
				30,
				31,
				32
			],
			outputs: [
				37,
				38,
				39,
				40
			]
		},
		{
			wgsl: stage10_default,
			inputs: [
				33,
				34,
				35,
				36,
				37,
				38,
				39,
				40
			],
			outputs: [
				41,
				42,
				43,
				44
			]
		},
		{
			wgsl: stage11_default,
			inputs: [
				33,
				34,
				35,
				36,
				37,
				38,
				39,
				40
			],
			outputs: [
				45,
				46,
				47,
				48
			]
		},
		{
			wgsl: stage12_default,
			inputs: [
				41,
				1,
				42,
				2,
				43,
				3,
				44,
				4,
				45,
				5,
				46,
				6,
				47,
				7,
				48,
				8
			],
			outputs: [49]
		},
		{
			wgsl: stage13_default,
			inputs: [49],
			outputs: [],
			final: true
		}
	]
};
//#endregion
export { model as default };
