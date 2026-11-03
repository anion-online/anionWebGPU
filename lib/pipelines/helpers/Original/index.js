
//#region src/pipelines/helpers/Original/index.ts
/**
* Original pipeline
*    Output the input texture without any change.
*/
var Original = class {
	outputTexture;
	/**
	* Creates an instance of Original.
	*
	* @param {Object} options - The options for the Original pipeline.
	* @param {GPUTexture} options.inputTexture - The input texture for the pipeline.
	*/
	constructor({ inputTexture }) {
		this.outputTexture = inputTexture;
	}
	updateParam(param, value) {
		throw new Error("Method not implemented.");
	}
	pass(encoder) {}
	getOutputTexture() {
		return this.outputTexture;
	}
};
//#endregion
export { Original };
