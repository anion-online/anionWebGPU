"""Выходы учителя на чистом входе — заранее, для `train.py --teacher-cache`.

    python teacher_cache.py data/init/2x_AnimeJaNai_V2_Compact_36k.onnx train128 teacher-compact.npy
    python teacher_cache.py data/teachers/apisr-rrdb/model.pth train128 teacher-apisr.npy
    python teacher_cache.py blend train128 teacher-compact.npy teacher-apisr.npy 0.75 teacher-compact-apisr075.npy
    python teacher_cache.py runs/u48r-compact/it80000.pt train128 teacher-u48-2.npy
    python teacher_cache.py runs/u48r-compact/it80000.pt train128 teacher-u48-2-lq.npy lq
    python teacher_cache.py runs/u48r-csdaulq/it100000.pt train128 teacher-u48-3-restore.npy lq-restore
    python teacher_cache.py ink train128 teacher-c-sd3-a075k5.npy teacher-u48-2.npy teacher-c-sd3-a075k5-u.npy
    python teacher_cache.py data/teachers/sharp4/model.safetensors anilibria/train teacher-rcan.npy clean

Цель обучения с `--teacher 1` — учитель на эталоне, уменьшенном 2×2
(`clean_input`). AnimeJaNai V2 Compact (64nf/16nc) в 6.5 раза тяжелее SUC —
считать его на каждом шаге значит учить вдвое дольше. Выход хранится как
uint8 BGR той же формы, что `gt.npy` (округление до уровня — много меньше
ошибки ученика).

Учитель — ONNX SRVGGNetCompact, `.pth`/`.safetensors` любой архитектуры ×2,
которую знает spandrel (как в `teacher_eval.py`), или чекпойнт `.pt` своей
сети (`load_any`).

Последний аргумент `lq` — учитель на порченом входе пары (`lq.npy`), как
ученик и как на Kodik; `lq-restore` — то же, но вход сперва проходит Restore
CNNM из Anime4K (Э23: цель «U48-3 после Restore» для дистилляции): так считаются линии U48-2 для `ink` — на чистом входе
она рисует штрих тоньше, чем может повторить ученик (Э20). `clean` — учитель
на готовом чистом входе `clean.npy` (пары AniLibria, Э29: кадр 1080p↓area до
720p; эталон там в 1.5×, а не ×2, и `clean_input` к нему не подходит).

`blend` складывает цель из двух кэшей (Э20): низ первого + верх второго —
`lo(a) + k·(b − lo(b))`, lo — гаусс σ в пикселях выхода. Так от APISR берутся
только резкие края линий (σ0.75), а цвет и заливки — от основного учителя:
у APISR целиком кожа светлеет и у контуров светлая кайма.

`ink` кладёт поверх цели линии другого кэша (U48-2: у неё штрих плотнее и
глаже, чем у составной цели) — `ink_overlay` из `teacher_eval.py`, та же
маска, что у листов `combos …^модель`.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from anime4k import Anime4K
from teacher_eval import ink_overlay
from train import DATA, clean_input, load_any, onnx_compact, to_tensor

BATCH = 64


def load_teacher(path: Path) -> torch.nn.Module:
    if path.suffix in (".pth", ".safetensors"):
        import spandrel

        desc = spandrel.ModelLoader().load_from_file(str(path))
        if desc.scale != 2:
            raise SystemExit(f"{path}: масштаб ×{desc.scale}, нужен ×2")
        return desc.model
    if path.suffix == ".pt":
        return load_any(path)
    return onnx_compact(path)


def blend(data: str, base: str, detail: str, sigma: float, name: str, k: float) -> None:
    a_all = np.load(DATA / data / base, mmap_mode="r")
    b_all = np.load(DATA / data / detail, mmap_mode="r")
    out = np.lib.format.open_memmap(DATA / data / name, "w+", np.uint8, a_all.shape)
    for i in range(len(a_all)):
        a, b = a_all[i].astype(np.float32), b_all[i].astype(np.float32)
        lo_a, lo_b = cv2.GaussianBlur(a, (0, 0), sigma), cv2.GaussianBlur(b, (0, 0), sigma)
        out[i] = np.clip(lo_a + k * (b - lo_b) + 0.5, 0, 255).astype(np.uint8)
        if i % 2000 == 0:
            print(f"{i}/{len(a_all)}", flush=True)
    out.flush()
    print(f"готово: {DATA / data / name}")


def ink(data: str, base: str, lines: str, name: str) -> None:
    a_all = np.load(DATA / data / base, mmap_mode="r")
    b_all = np.load(DATA / data / lines, mmap_mode="r")
    out = np.lib.format.open_memmap(DATA / data / name, "w+", np.uint8, a_all.shape)
    for i in range(len(a_all)):
        mixed = ink_overlay(a_all[i].astype(np.float32), b_all[i].astype(np.float32))
        out[i] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)
        if i % 2000 == 0:
            print(f"{i}/{len(a_all)}", flush=True)
    out.flush()
    print(f"готово: {DATA / data / name}")


def main() -> None:
    if sys.argv[1] == "ink":
        ink(*sys.argv[2:])
        return
    if sys.argv[1] == "blend":
        data, base, detail, sigma, name, *k = sys.argv[2:]
        blend(data, base, detail, float(sigma), name, float(k[0]) if k else 1.0)
        return
    model_path, data, name, *source = sys.argv[1:]
    from_lq = source in (["lq"], ["lq-restore"])
    from_clean = source == ["clean"]
    restore = Anime4K("Anime4K_Restore_CNN_M.glsl") if source == ["lq-restore"] else None
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    teacher = load_teacher(Path(model_path)).to(device).eval()
    restore = restore.to(device) if restore is not None else None
    gt_all = np.load(DATA / data / "gt.npy", mmap_mode="r")
    src_all = np.load(DATA / data / ("lq.npy" if from_lq else "clean.npy"), mmap_mode="r") \
        if from_lq or from_clean else gt_all
    n, h, w, c = src_all.shape
    shape = (n, 2 * h, 2 * w, c) if from_lq or from_clean else gt_all.shape
    out = np.lib.format.open_memmap(DATA / data / name, "w+", np.uint8, shape)
    with torch.no_grad():
        for start in range(0, len(gt_all), BATCH):
            src = to_tensor(src_all[start : start + BATCH], device)
            if restore is not None:
                src = restore(src).clamp(0, 1)
            rgb = teacher(src if from_lq or from_clean else clean_input(src)).clamp(0, 1).permute(0, 2, 3, 1)
            out[start : start + len(src)] = (rgb * 255 + 0.5).byte().cpu().numpy()[..., ::-1]
            if start // BATCH % 50 == 0:
                print(f"{start}/{len(gt_all)}", flush=True)
    out.flush()
    print(f"готово: {DATA / data / name}")


if __name__ == "__main__":
    main()
