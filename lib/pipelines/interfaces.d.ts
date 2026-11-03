/// <reference types="@webgpu/types" />
import { FrameGate } from "./helpers/FrameGate/index.js";
//#region src/pipelines/interfaces.d.ts
export interface Anime4KPipeline {
  /**
   * Update the controllable parameter managed by the pipeline
   *
   * @param param  - name of the parameter
   * @param value  - value of the parameter
   */
  updateParam(param: string, value: any): void;
  /**
   * write compute commands into the encoder
   *
   * @param encoder - encoder to record commands into
   */
  pass(encoder: GPUCommandEncoder): void;
  /**
   * get the output texture of this pipeline
   */
  getOutputTexture(): GPUTexture;
  /**
   * Шейдеры скомпилированы. Звенья форка собирают пайплайны через
   * createComputePipelineAsync — компиляция идёт в фоне и не замораживает
   * страницу; до готовности pass() бросает ошибку. Звенья без поля готовы
   * сразу (синхронная сборка).
   */
  ready?: Promise<void>;
}
/** Готовность цепочки: все звенья, у которых есть ready. */
export declare function whenReady(pipelines: Anime4KPipeline[]): Promise<void>;
export interface OriginalPipelineDescriptor {
  inputTexture: GPUTexture;
}
export interface Conv2dPipelineDescriptor {
  device: GPUDevice;
  inputTextures: GPUTexture[];
  shaderWGSL: string;
  name?: string;
}
export interface DepthToSpacePipelineDescriptor {
  device: GPUDevice;
  inputTextures: GPUTexture[];
  name?: string;
}
export interface OverlayPipelineDescriptor {
  device: GPUDevice;
  inputTextures: GPUTexture[];
  outputTextureSize: number[];
  fragmentWGSL?: string;
  name?: string;
}
export interface DownscalePipelineDescriptor {
  device: GPUDevice;
  inputTexture: GPUTexture;
  targetDimensions: {
    width: number;
    height: number;
  };
  /** По умолчанию 'catmull-rom' (без алиасинга); 'bilinear' — как в mpv. */
  filter?: 'catmull-rom' | 'bilinear';
  /** Статистика ClampStats: зажим ореолов прямо в последнем проходе. */
  deRing?: GPUTexture;
  /** Ворота повторов: проходы запускаются косвенно и на повторе пропускаются. */
  gate?: FrameGate;
  name?: string;
}
export interface ClampHighlightsPipelineDescriptor {
  device: GPUDevice;
  /** Результат цепочки — его и зажимаем, в его разрешении. */
  inputTexture: GPUTexture;
  /** Исходный кадр: по нему считается максимум яркости 5×5. */
  statsTexture: GPUTexture;
  name?: string;
}
export interface Anime4KPipelineDescriptor extends OriginalPipelineDescriptor {
  device: GPUDevice;
}
/**
 * Точность арифметики CNN-моделей. 'f16' требует устройства с фичей
 * 'shader-f16'. По умолчанию f32: на Apple f16 не быстрее (та же скорость
 * ALU), на GPU с двойной скоростью f16 (Intel, AMD) выигрыш нужно подтвердить
 * замером. Точность f16 против f32: maxΔ 0.81/255, PSNR 70 дБ.
 */
export type CNNPrecision = 'f32' | 'f16';
export interface CNNModelPipelineDescriptor extends Anime4KPipelineDescriptor {
  precision?: CNNPrecision;
  /** Статистика ClampStats: зажим ореолов прямо в финальной стадии модели. */
  deRing?: GPUTexture;
  /** Ворота повторов: стадии запускаются косвенно и на повторе пропускаются. */
  gate?: FrameGate;
}
export interface Anime4KPresetPipelineDescriptor extends Anime4KPipelineDescriptor {
  nativeDimensions: {
    width: number;
    height: number;
  };
  targetDimensions: {
    width: number;
    height: number;
  };
}
/** Размер модели Upscale-Denoise: M ≈ в 4 раза дешевле VL, L — посередине. */
export type DenoiseModelSize = 'M' | 'L' | 'VL';
export interface ModeCPresetPipelineDescriptor extends Anime4KPresetPipelineDescriptor {
  /** По умолчанию VL — как в Anime4K Mode C. */
  denoiseModel?: DenoiseModelSize;
  /** Точность CNN-звеньев, по умолчанию f32. */
  precision?: CNNPrecision;
  /**
   * Не пересчитывать повторяющиеся кадры (FrameGate): на повторе цепочка не
   * запускается, выход остаётся прошлым. По умолчанию выключено.
   */
  skipUnchanged?: boolean;
  /**
   * Порог повтора в уровнях 8 бит, по умолчанию 0 — только точные повторы,
   * выход тот же, что без пропуска.
   */
  unchangedThreshold?: number;
}
export interface ModeArtCNNPresetPipelineDescriptor extends Anime4KPresetPipelineDescriptor {
  /** C4F16 (~12k параметров, по умолчанию) или C4F32 (~48k). */
  model?: 'C4F16' | 'C4F32';
  precision?: CNNPrecision;
}
//#endregion