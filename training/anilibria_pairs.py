"""Пары Э28: эталон — кадр AniLibria 1080p, вход — он же в 720p с порчей Kodik.

    python anilibria_pairs.py [--frames 80] [--only <slug>] [--keep]
    python anilibria_pairs.py --pack [--crops 8]   # фрагменты для train.py --al

Серии — в EPISODES. Для каждой: оба потока (1080p и 720p) качаются целиком
одним запросом (CDN отвечает 429 на параллельные), дальше всё локально, и
после серии файлы удаляются (`--keep` — оставить). Вокруг каждого кадра —
клип 3 с; из него варианты входа, каждый — x264 с настройками Kodik
(`degrade.X264_PARAMS`), два прохода в случайный битрейт 800–1300 кбит/с:

- `area` / `bicubic` — 1080p, уменьшенный до 720p (ядро — случайно);
- `re720` — 720p самой AniLibria, пережатый Kodik (двойное сжатие: энкод
  группы, потом Kodik — так у перекодированных Kodik озвучек).

Замер Э28 (2026-11-10, кадры `goblin1`): настоящий Kodik к 1080↓area —
43.9 дБ без логотипа; area 46.3, bicubic 45.2, re720 44.3.

Выход — `data/anilibria/<slug>/`: `gt/<t>.png` (1080p), `lq/<t>_<вариант>.png`
(720p), `meta.jsonl` — по строке на вариант (битрейт, PSNR к 1080↓area,
доля деталей выше 720p у эталона). Кадр, у которого вариант разошёлся с
эталоном (PSNR < 30 — сдвиг потоков или смена плана), не сохраняется.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from degrade import X264_PARAMS

DATA = Path(__file__).parent / "data" / "anilibria"
API = "https://aniliberty.top/api/v1/anime/releases/{}"
UA = {"User-Agent": "Mozilla/5.0"}
FPS = 24000 / 1001
CLIP = 3.0  # с; кадр — на 1-й секунде клипа
AT = 24  # номер кадра в клипе (≈1.001 с)
BITRATE = (800, 1300)  # кбит/с

# (slug, релиз AniLiberty, серия, назначение). val — целые кадры для проверки.
EPISODES = [
    ("goblin-slayer-2-e02", 9537, 2, "train"),
    ("solo-leveling-e03", 9600, 3, "train"),
    ("spy-family-s2-e02", 9535, 2, "train"),
    ("oshi-no-ko-2-e02", 9724, 2, "train"),
    ("bocchi-e03", 9293, 3, "train"),
    ("shangri-la-e02", 9562, 2, "train"),
    ("sakamoto-days-e02", 9886, 2, "train"),
    ("witch-watch-e02", 9916, 2, "train"),
    ("dr-stone-3-e02", 9407, 2, "train"),
    ("vinland-2-e02", 9327, 2, "train"),
    ("mashle-2-e02", 9602, 2, "train"),
    ("dungeon-meshi-e02", 9612, 2, "train"),
    ("lycoris-e02", 9233, 2, "train"),
    ("kaijuu-8-s2-e02", 9963, 2, "train"),
    ("dandadan-e02", 9789, 2, "train"),
    ("goblin-reborn-e01", 10338, 1, "val"),
    ("frieren-2-e01", 10085, 1, "val"),
]


def log(text: str) -> None:
    print(time.strftime("%H:%M:%S"), text, flush=True)


def episode_info(release: int, number: int) -> dict:
    req = urllib.request.Request(API.format(release), headers=UA)
    data = json.load(urllib.request.urlopen(req, timeout=30))
    ep = next(e for e in data["episodes"] if int(e["ordinal"]) == number)
    if not ep.get("hls_1080") or not ep.get("hls_720"):
        raise SystemExit(f"релиз {release} серия {number}: нет 1080p или 720p")
    return ep


def download(url: str, out: Path) -> None:
    if out.exists() and out.stat().st_size > 0:
        return
    tmp = out.with_suffix(".part")
    for attempt in range(4):
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-user_agent", UA["User-Agent"],
                            "-i", url, "-map", "0:v:0", "-c", "copy", "-y", "-f", "mpegts", str(tmp)])
        if r.returncode == 0 and tmp.stat().st_size > 0:
            tmp.rename(out)
            return
        time.sleep(10 * (attempt + 1))
    raise SystemExit(f"не скачалось: {url}")


def pick_times(ep: dict, n: int, rng: np.random.Generator) -> list[float]:
    """Моменты кадров: равномерно по серии со случайным сдвигом, мимо опенинга
    и эндинга (разметка AniLiberty) и первых/последних 20 с."""
    dur = float(ep["duration"])
    skip = [(s.get("start"), s.get("stop")) for s in (ep.get("opening") or {}, ep.get("ending") or {})]
    skip = [(a - 3, b + 3) for a, b in skip if a is not None and b is not None]
    if not (ep.get("ending") or {}).get("start"):
        skip.append((dur - 100, dur))  # эндинг без разметки — последние ~1.5 мин
    lo, hi = 20.0, dur - 20.0
    grid = np.linspace(lo, hi, n * 3)
    ok = [t for t in grid if all(not (a <= t <= b) for a, b in skip)]
    idx = np.unique(np.linspace(0, len(ok) - 1, min(n, len(ok))).round().astype(int))
    jitter = (hi - lo) / (n * 3)
    return [float(ok[i] + rng.uniform(-0.4, 0.4) * jitter) for i in idx]


def decode_clip(src: Path, start: float, w: int, h: int) -> np.ndarray:
    # Потоки AniLibria — с переменной частотой (повторы кадров выброшены, и
    # в 1080p и 720p по-разному): номер кадра в клипе не равен времени.
    # fps= приводит к постоянной 23.976 — кадр AT в обоих потоках один момент.
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src),
                          "-t", f"{CLIP}", "-an", "-vf", "fps=24000/1001:start_time=0", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def kodik_encode(frames: np.ndarray, kbps: int, work: Path) -> np.ndarray:
    """Клип 720p → x264 Kodik, два прохода → декодированный кадр AT."""
    h, w = frames.shape[1:3]
    src = work / "clip.yuv"
    yuv = np.concatenate([cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420).ravel() for f in frames])
    yuv.tofile(src)
    common = ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p",
              "-s", f"{w}x{h}", "-r", "24000/1001", "-i", str(src), "-c:v", "libx264", "-preset", "medium",
              "-b:v", f"{kbps}k", "-x264-params", X264_PARAMS, "-passlogfile", str(work / "pass")]
    subprocess.run(common + ["-pass", "1", "-f", "null", "-"], check=True)
    subprocess.run(common + ["-pass", "2", "-y", str(work / "clip.mp4")], check=True)
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(work / "clip.mp4"),
                          "-vf", f"select=eq(n\\,{AT})", "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def luma(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray, c: int = 8) -> float:
    mse = float(np.mean((luma(a)[c:-c, c:-c] - luma(b)[c:-c, c:-c]) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def aligned(a: np.ndarray, b: np.ndarray) -> bool:
    """Тот же ли момент: PSNR размытых копий (σ 2) ≥ 30. Без размытия сильное
    сжатие (быстрый фон в 720p AniLibria — 24 дБ) выглядело бы как сдвиг."""
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), 2)  # noqa: E731
    return psnr(blur(a), blur(b)) >= 30


def above_720(gt: np.ndarray) -> float:
    """PSNR эталона к нему же через 720p: выше — меньше деталей сверх 720p
    (растянутый 720p даёт ~50+)."""
    back = cv2.resize(cv2.resize(gt, (1280, 720), interpolation=cv2.INTER_AREA), (1920, 1080),
                      interpolation=cv2.INTER_LANCZOS4)
    return psnr(gt, back, 16)


def process(slug: str, release: int, number: int, role: str, frames: int, keep: bool) -> None:
    out = DATA / slug
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "lq").mkdir(exist_ok=True)
    work = DATA / ".work"
    work.mkdir(exist_ok=True)
    done = {json.loads(line)["t"] for line in (out / "meta.jsonl").open()} if (out / "meta.jsonl").exists() else set()

    ep = episode_info(release, number)
    t0 = time.time()
    src = {q: DATA / f".{slug}-{q}.ts" for q in (1080, 720)}
    for q in (1080, 720):
        download(ep[f"hls_{q}"], src[q])
    log(f"{slug}: скачано за {time.time() - t0:.0f} с "
        f"({src[1080].stat().st_size >> 20} + {src[720].stat().st_size >> 20} МБ)")

    rng = np.random.default_rng(release * 1000 + number)
    kept = 0
    with (out / "meta.jsonl").open("a") as meta:
        for t in pick_times(ep, frames, rng):
            name = f"{t:07.2f}"
            if name in done:
                continue
            start = max(0.0, t - AT / FPS)
            hd = decode_clip(src[1080], start, 1920, 1080)
            sd = decode_clip(src[720], start, 1280, 720)
            if len(hd) <= AT + 8 or len(sd) <= AT + 8:
                continue
            gt = hd[AT]
            y = luma(gt)
            if y.mean() < 12 or y.std() < 6:  # чёрный кадр или заливка
                continue
            ref = cv2.resize(gt, (1280, 720), interpolation=cv2.INTER_AREA)
            if not aligned(sd[AT], ref):  # потоки разошлись
                log(f"  {name}: 720p не совпал с 1080p, пропуск")
                continue
            kernel = "area" if rng.random() < 0.5 else "bicubic"
            flag = cv2.INTER_AREA if kernel == "area" else cv2.INTER_CUBIC
            variants = {
                kernel: np.stack([cv2.resize(f, (1280, 720), interpolation=flag) for f in hd]),
                "re720": sd,
            }
            rows = []
            for vname, clip in variants.items():
                kbps = int(rng.integers(*BITRATE))
                lq = kodik_encode(clip, kbps, work)
                if not aligned(lq, ref):
                    rows = []
                    break
                p = psnr(lq, ref)
                rows.append((vname, lq, {"t": name, "variant": vname, "kbps": kbps, "psnr": round(p, 2)}))
            if not rows:
                log(f"  {name}: вариант разошёлся с эталоном, пропуск")
                continue
            cv2.imwrite(str(out / "gt" / f"{name}.png"), gt)
            hf = round(above_720(gt), 2)
            for vname, lq, row in rows:
                cv2.imwrite(str(out / "lq" / f"{name}_{vname}.png"), lq)
                meta.write(json.dumps(row | {"above720": hf, "role": role}) + "\n")
            meta.flush()
            kept += 1
    for f in work.iterdir():
        f.unlink()
    if not keep:
        for f in src.values():
            f.unlink(missing_ok=True)
    log(f"{slug}: кадров {kept}, {time.time() - t0:.0f} с")


PATCH = 128  # вход фрагмента, 720p; эталон — PATCH * 3 // 2 из 1080p
VAL_PATCH = 256
# Углы сверху, где у озвучек логотипы (в координатах 720p): фрагменты их не задевают.
LOGO = (320, 128)


def crop_xy(rng: np.random.Generator, size: int) -> tuple[int, int]:
    """Случайный угол фрагмента 720p: чётный (эталон — ровно ×1.5) и мимо логотипов."""
    while True:
        x = int(rng.integers(0, (1280 - size) // 2 + 1)) * 2
        y = int(rng.integers(0, (720 - size) // 2 + 1)) * 2
        in_corner = y < LOGO[1] and (x < LOGO[0] or x + size > 1280 - LOGO[0])
        if not in_corner:
            return x, y


def pack(crops: int, val_crops: int) -> None:
    """Фрагменты из всех серий в `data/anilibria/{train,val}/{lq,gt}.npy` —
    как `train128`, но эталон в 1.5× входа (128 → 192, проверка 256 → 384).
    `clean.npy` — эталон, уменьшенный area до входа: чистый 720p для учителей
    (`teacher_cache.py … clean`, Э29). Угол эталона кратен 3, так что это
    ровно фрагмент кадра 1080p↓area."""
    role = {slug: r for slug, _, _, r in EPISODES}
    jobs: dict[str, list[tuple[Path, Path]]] = {"train": [], "val": []}
    for slug in sorted(role):
        for lq in sorted((DATA / slug / "lq").glob("*.png")):
            jobs[role[slug]].append((lq, DATA / slug / "gt" / (lq.stem.rsplit("_", 1)[0] + ".png")))
    for part, n, size in (("train", crops, PATCH), ("val", val_crops, VAL_PATCH)):
        out = DATA / part
        out.mkdir(exist_ok=True)
        big = size * 3 // 2
        count = len(jobs[part]) * n
        lq_out = np.lib.format.open_memmap(out / "lq.npy", "w+", np.uint8, (count, size, size, 3))
        gt_out = np.lib.format.open_memmap(out / "gt.npy", "w+", np.uint8, (count, big, big, 3))
        clean_out = np.lib.format.open_memmap(out / "clean.npy", "w+", np.uint8, (count, size, size, 3))
        rng = np.random.default_rng(0 if part == "train" else 1)
        i = 0
        for k, (lq_path, gt_path) in enumerate(jobs[part]):
            lq, gt = cv2.imread(str(lq_path)), cv2.imread(str(gt_path))
            assert lq.shape == (720, 1280, 3) and gt.shape == (1080, 1920, 3), lq_path
            for _ in range(n):
                x, y = crop_xy(rng, size)
                lq_out[i] = lq[y : y + size, x : x + size]
                X, Y = x * 3 // 2, y * 3 // 2
                gt_out[i] = gt[Y : Y + big, X : X + big]
                clean_out[i] = cv2.resize(gt_out[i], (size, size), interpolation=cv2.INTER_AREA)
                i += 1
            if k % 500 == 0:
                print(f"{part} {k}/{len(jobs[part])}", flush=True)
        lq_out.flush()
        gt_out.flush()
        clean_out.flush()
        print(f"{part}: {count} пар {size}→{big} из {len(jobs[part])} кадров-вариантов", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=80, help="кадров на серию")
    p.add_argument("--only", help="одна серия по slug")
    p.add_argument("--keep", action="store_true", help="не удалять скачанные потоки")
    p.add_argument("--pack", action="store_true", help="не качать, а нарезать фрагменты для train.py --al")
    p.add_argument("--crops", type=int, default=8, help="--pack: фрагментов на кадр-вариант")
    p.add_argument("--val-crops", type=int, default=2, help="--pack: фрагментов проверки на кадр-вариант")
    args = p.parse_args()
    if args.pack:
        pack(args.crops, args.val_crops)
        return
    todo = [e for e in EPISODES if args.only in (None, e[0])]
    if not todo:
        sys.exit(f"нет серии {args.only}")
    for slug, release, number, role in todo:
        process(slug, release, number, role, args.frames, args.keep)


if __name__ == "__main__":
    main()
