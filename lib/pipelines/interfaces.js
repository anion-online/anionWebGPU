
//#region src/pipelines/interfaces.ts
/** Готовность цепочки: все звенья, у которых есть ready. */
function whenReady(pipelines) {
	return Promise.all(pipelines.map((pipeline) => pipeline.ready)).then(() => void 0);
}
//#endregion
export { whenReady };
