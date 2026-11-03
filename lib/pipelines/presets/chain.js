
//#region src/pipelines/presets/chain.ts
/**
* Решения те же, что в Anime4K Mode C (условия //!WHEN в GLSL): считаются до
* постройки, чтобы знать последнее звено — ему отдаётся зажим Clamp
* Highlights (deRing) вместо отдельного прохода.
*/
function planModeC(native, target) {
	const above = (k) => target.width > k * native.width && target.height > k * native.height;
	const below = (k) => target.width < k * native.width && target.height < k * native.height;
	const upscale1 = above(1.2);
	const downscale2 = above(1.2) && below(2);
	const downscale4 = above(2.4) && below(4);
	let current = native;
	if (upscale1) current = {
		width: native.width * 2,
		height: native.height * 2
	};
	if (downscale2) current = target;
	if (downscale4) current = {
		width: Math.ceil(target.width / 2),
		height: Math.ceil(target.height / 2)
	};
	return {
		upscale1,
		downscale2,
		downscale4,
		upscale2: target.width > 1.2 * current.width && target.height > 1.2 * current.height
	};
}
//#endregion
export { planModeC };
