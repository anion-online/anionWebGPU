/// <reference types="@webgpu/types" />
import { Anime4KPipeline } from "../../interfaces.js";
//#region src/pipelines/helpers/FrameGate/index.d.ts
export interface FrameGateDescriptor {
  device: GPUDevice;
  /** Исходный кадр цепочки — его и сравниваем с прошлым. */
  inputTexture: GPUTexture;
  /**
   * Порог в уровнях 8 бит: кадр пропускается, если ни один канал ни одного
   * пикселя не отличается от последнего посчитанного больше чем на него.
   * 0 (по умолчанию) — только точные повторы: выход тот же, что без ворот.
   * У видео после x264 точных повторов почти нет: тот же рисунок отличается
   * шумом сжатия до ~16 уровней (docs/PLAN.md, Э15).
   */
  threshold?: number;
  /**
   * Второе условие пропуска: средний по кадру знаковый сдвиг каждого канала
   * не больше этого числа уровней, по умолчанию 0.5. Шум сжатия в среднем
   * около нуля, а затемнение с малым шагом — нет: без условия оно шло бы
   * ступенями по `threshold`. При `threshold` 0 ничего не меняет.
   */
  meanThreshold?: number;
  name?: string;
}
/** Запуск звена: напрямую или косвенно через ворота. */
export interface Launch {
  dispatch(pass: GPUComputePassEncoder): void;
}
/** Счётчики ворот с момента сборки. */
export interface FrameGateStats {
  frames: number;
  skipped: number;
  /** max |Δ| последнего кадра с опорным, в уровнях 8 бит. */
  lastDiff: number;
}
/**
 * Запуск звена с `gate` — косвенный из буфера ворот, без него — как раньше.
 * Число групп известно при сборке звена, поэтому слот берётся там же.
 */
export declare function launcher(gate: FrameGate | undefined, x: number, y: number): Launch;
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
export declare class FrameGate implements Anime4KPipeline {
  name: string;
  ready: Promise<void>;
  /** Аргументы косвенного запуска звеньев (INDIRECT). */
  readonly args: GPUBuffer;
  private device;
  private template;
  private state;
  private readback?;
  private reading?;
  private prev;
  private input;
  private slots;
  private width;
  private height;
  private pipelines;
  private bindGroups;
  private copyLaunch;
  constructor({ device, inputTexture, threshold, meanThreshold, name }: FrameGateDescriptor);
  /** Слот косвенного запуска на `x × y` групп; звено зовёт при сборке. */
  slot(x: number, y: number): Launch;
  /** Порог в уровнях 8 бит (см. `threshold`). */
  setThreshold(threshold: number): void;
  /** Порог среднего сдвига в уровнях 8 бит (см. `meanThreshold`). */
  setMeanThreshold(meanThreshold: number): void;
  /** Посчитать следующий кадр, даже если он повтор. */
  invalidate(): void;
  /** Счётчики с GPU; параллельные вызовы получают одно чтение. */
  readStats(): Promise<FrameGateStats>;
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  /** Ворота ничего не выводят — отдают вход как есть. */
  getOutputTexture(): GPUTexture;
}
//#endregion