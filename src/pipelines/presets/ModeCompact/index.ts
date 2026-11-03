import { Downscale, FrameGate } from '../../helpers';
import { Deband, DebandParams } from '../../denoise/Deband';
import { Anime4KPipeline, Anime4KPresetPipelineDescriptor, whenReady } from '../../interfaces';
import { CompactKernel, CompactModelData, CompactSR } from '../../compact';
import { CNNM } from '../../restore';

export interface ModeCompactPresetPipelineDescriptor extends Anime4KPresetPipelineDescriptor {
  /** Веса и описание из конвертера anion-dl; в пакет не входят (лицензия NC). */
  model: CompactModelData;
  /** Ядро, по умолчанию `fast`. */
  kernel?: CompactKernel;
  /**
   * Не пересчитывать повторяющиеся кадры (FrameGate): на повторе сеть не
   * запускается, выход остаётся прошлым. По умолчанию выключено.
   */
  skipUnchanged?: boolean;
  /** Порог повтора в уровнях 8 бит, по умолчанию 0 — только точные повторы. */
  unchangedThreshold?: number;
  /**
   * Цветность выхода: σ сглаживания цветности кадра, по умолчанию 2 — от
   * сети берётся только яркость (цветовые блоки сжатия сеть рисует пятнами).
   * 0 — цветность сети, как до Э12. См. CompactPipelineDescriptor.
   */
  chromaSigma?: number;
  /**
   * Порог сглаживания цветности (Э27): по умолчанию 0 — как в 1.x. 0.02 —
   * цветные линии не расплываются. См. CompactPipelineDescriptor.
   */
  chromaThreshold?: number;
  /**
   * Дебандинг до сети (Э21): `true` — с параметрами по умолчанию, объект —
   * со своими. По умолчанию выключен: выход тот же, что в 1.x.
   */
  deband?: boolean | DebandParams;
  /**
   * Restore CNNM из «С очисткой» (Э23) — снятие размытия, текст и линии
   * плотнее: `before` — на входе сети, `after` — в разрешении цели, как в
   * ModeCA. По умолчанию нет.
   */
  restore?: 'before' | 'after';
}

/**
 * «Детали»: SRVGGNetCompact ×2 (AnimeJaNai V2 SuperUltraCompact), затем
 * уменьшение до цели, если она меньше 2× (Catmull-Rom). При цели ≤ 1.2×
 * исходника не делает ничего — как ModeArtCNN.
 */
export class ModeCompact implements Anime4KPipeline {
  pipelines: Anime4KPipeline[] = [];

  outputTexture: GPUTexture;

  ready: Promise<void>;

  /** Ворота повторов (skipUnchanged): счётчики — gate.readStats(). */
  gate?: FrameGate;

  /** Дебандинг (deband): параметры меняются на ходу — deband.set(). */
  deband?: Deband;

  constructor({
    device, inputTexture, nativeDimensions, targetDimensions, model, kernel,
    skipUnchanged = false, unchangedThreshold = 0, chromaSigma = 2, chromaThreshold = 0, deband, restore,
  }: ModeCompactPresetPipelineDescriptor) {
    let currentTexture = inputTexture;
    if (targetDimensions.width > 1.2 * nativeDimensions.width
        && targetDimensions.height > 1.2 * nativeDimensions.height) {
      // Ворота — первыми: остальные звенья запускаются из их буфера.
      const gate = skipUnchanged
        ? new FrameGate({ device, inputTexture, threshold: unchangedThreshold })
        : undefined;
      if (gate) this.pipelines.push(gate);
      this.gate = gate;

      // До сети: ступенька градиента иначе прорисовывается как край. Ворота
      // сравнивают исходный кадр — дебандинг на их решение не влияет.
      if (deband) {
        this.deband = new Deband({
          device, inputTexture: currentTexture, gate, ...(deband === true ? {} : deband),
        });
        this.pipelines.push(this.deband);
        currentTexture = this.deband.getOutputTexture();
      }

      if (restore === 'before') {
        const cnnm = new CNNM({ device, inputTexture: currentTexture, gate });
        this.pipelines.push(cnnm);
        currentTexture = cnnm.getOutputTexture();
      }

      const upscale = new CompactSR({
        device, inputTexture: currentTexture, model, kernel, gate, chromaSigma, chromaThreshold,
      });
      this.pipelines.push(upscale);
      currentTexture = upscale.getOutputTexture();

      if (targetDimensions.width < 2 * nativeDimensions.width
          && targetDimensions.height < 2 * nativeDimensions.height) {
        const downscale = new Downscale({
          device, inputTexture: currentTexture, targetDimensions, gate,
        });
        this.pipelines.push(downscale);
        currentTexture = downscale.getOutputTexture();
      }

      if (restore === 'after') {
        const cnnm = new CNNM({ device, inputTexture: currentTexture, gate });
        this.pipelines.push(cnnm);
        currentTexture = cnnm.getOutputTexture();
      }
    }
    this.outputTexture = currentTexture;
    this.ready = whenReady(this.pipelines);
  }

  updateParam(param: string, value: any): void {
    throw new Error('Preset has no param');
  }

  pass(encoder: GPUCommandEncoder): void {
    this.pipelines.forEach((pipeline) => pipeline.pass(encoder));
  }

  getOutputTexture(): GPUTexture {
    return this.outputTexture;
  }
}
