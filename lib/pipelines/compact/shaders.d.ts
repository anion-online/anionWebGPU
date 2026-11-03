/// <reference types="@webgpu/types" />
//#region src/pipelines/compact/shaders.d.ts
/**
 * WGSL для SRVGGNetCompact («Детали» в anion-dl, AnimeJaNai V2
 * SuperUltraCompact): 10 свёрток 3×3 с PReLU в разрешении источника,
 * увеличение — pixelshuffle последнего слоя.
 *
 * Ядра (`CompactKernel`):
 * - `reference` — перенос ядра anion-dl (`src/app/player/compact/shaders.ts`,
 *   сверено там с onnxruntime на 72.6 дБ): по пикселю на поток, 24
 *   накопителя f32, `dot` на vec4<f16>, веса из storage-буфера. Эталон для
 *   сверки быстрых ядер.
 * - `fast` — слой как умножение mat4x4<f16> на vec4<f16>: накопители — vec4
 *   по четыре выходных канала, 16 FMA на 16 умножений, без переводов
 *   f16 → f32 на каждом слагаемом; поток — 2×2 пикселя × 3 выходные группы
 *   из 6 (вес читается раз на четыре пикселя), тайл `[группа][пиксель]`; веса слоя — отдельный storage-буфер `array<mat4x4<f16>>`: матрица
 *   читается одной выборкой с одной проверкой границ. Сумма целиком в f16.
 * - `fast-f32acc` — то же, но сумма по каждому отводу ядра (24 слагаемых)
 *   копится в f16 и переносится в накопитель f32: точнее, но медленнее.
 *
 * Замер (Chrome 153, M1 Pro, 1280×720, только сеть): reference 45.6 мс,
 * fast-f32acc 38.1 мс, fast 29.3 мс; раскладка 2×2 × 3 группы (Э14) — ещё
 * −9.5% у fast и −18% у fast-f32acc, выход бит в бит прежний. Проверены и отброшены (docs/PLAN.md, Э10):
 * веса литералами в коде (в 10 раз медленнее), mat4x4<f32>, веса в общей
 * памяти группы, 3–4 пикселя на поток, веса в uniform-буфере (на wgpu −8%, в
 * Chrome +50%: Tint раскладывает mat4x4<f16> в uniform по-своему).
 *
 * Отличия от anion-dl — только ввод-вывод: первый слой читает кадр прямо из
 * текстуры (без прохода препроцесса), последний пишет выход ×2 в текстуру
 * rgba16float, зажатый в 0..1.
 */
/**
 * `wino` — внутренние слои Winograd F(2×2, 3×3) (`winoLayerShader`), первый и
 * последний — `fast`. Быстрее `fast` только в WebKit (Safari 27: −12%), в
 * Chrome на 23% медленнее — выбор по движку на стороне плеера (Э18).
 */
export type CompactKernel = 'reference' | 'fast' | 'fast-f32acc' | 'wino';
/** Слой из `model.json` конвертера anion-dl; `offset` — в байтах от начала блоба. */
export interface CompactLayerMeta {
  name: string;
  offset: number;
  in: number;
  out: number;
  inGroups: number;
  outGroups: number;
  prelu: boolean;
}
/** `model.json` конвертера anion-dl (`scripts/compact-to-wgsl.py`). */
export interface CompactModelMeta {
  arch: string;
  numFeat: number;
  numConv: number;
  scale: number;
  bytes: number;
  layers: CompactLayerMeta[];
  /**
   * `SRVGGNetCompactU` (Э19): сеть считает на сетке вдвое реже кадра — вход
   * разложен pixel-unshuffle 2×2 (канал c·4 + dy·2 + dx), выход —
   * pixelshuffle ×(2·scale). Конвертер — `training/to_wgsl.py`.
   */
  unshuffle?: number;
  source?: string;
  license?: string;
}
//#endregion