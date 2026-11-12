"""Дообучение SRVGGNetCompact (SuperUltraCompact, 24nf/8nc/×2) на порче Kodik
и обучение с нуля своих сетей половинной сетки (CompactU, Э19).

    python train.py --init model.onnx --run runs/<имя> [--iters 30000]
    python train.py --arch u48x7r --run runs/<имя> --teacher 1 --ink 8 [--data train128]
    python train.py --arch u48x7 --from runs/<сеть>/it100000.pt … --text train128-text  # надписи, Э22
    python train.py --arch u48x7 --from runs/u48-rcan-ft/it25000.pt … --al anilibria  # 1080p, Э28

Стартует с весов ONNX (AnimeJaNai V2 SUC — то, что сейчас в «Деталях»),
учится на `data/train/{lq,gt}.npy` (`make_pairs.py`, грузятся в память), каждые `--every`
итераций считает PSNR на целых кадрах `data/val/` и пишет чекпойнт. В конце —
`model.onnx` той же формы, что исходная: её берёт `compact-to-wgsl.py convert`
из anion-dl.
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper
from torch import nn

DATA = Path(__file__).parent / "data"


class Compact(nn.Module):
    """SRVGGNetCompact из Real-ESRGAN: свёртки 3×3 с PReLU, PixelShuffle и
    прибавка входа, растянутого ближайшим соседом (как в `srvgg_arch.py`)."""

    def __init__(self, num_feat: int = 24, num_conv: int = 8, scale: int = 2):
        super().__init__()
        self.scale = scale
        layers: list[nn.Module] = [nn.Conv2d(3, num_feat, 3, padding=1), nn.PReLU(num_feat)]
        for _ in range(num_conv):
            layers += [nn.Conv2d(num_feat, num_feat, 3, padding=1), nn.PReLU(num_feat)]
        layers.append(nn.Conv2d(num_feat, 3 * scale * scale, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pixel_shuffle(self.body(x), self.scale)
        return out + F.interpolate(x, scale_factor=self.scale, mode="nearest")


class RepConv(nn.Module):
    """Свёртка 3×3 на время обучения — с параллельной веткой 1×1 и, с
    `identity`, тождеством, где каналов на входе и выходе поровну
    (перепараметризация, как RepVGG без BN). `fuse()` сливает ветки в одну
    Conv2d 3×3 с тем же выходом: ядро и цена кадра не меняются.

    Тождество в каждом внутреннем слое без нормировки раздувает сигнал: на
    старте u48x7 давала 18.6 дБ против 27 у обычной и к 10k итераций
    отставала на 0.5 дБ (Э19) — по умолчанию его нет."""

    def __init__(self, cin: int, cout: int, identity: bool):
        super().__init__()
        self.conv3 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv1 = nn.Conv2d(cin, cout, 1)
        self.identity = identity and cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3(x) + self.conv1(x)
        return y + x if self.identity else y

    def fuse(self) -> nn.Conv2d:
        conv = nn.Conv2d(self.conv3.in_channels, self.conv3.out_channels, 3, padding=1)
        with torch.no_grad():
            weight = self.conv3.weight.clone()
            weight[:, :, 1, 1] += self.conv1.weight[:, :, 0, 0]
            if self.identity:
                channels = torch.arange(weight.shape[0])
                weight[channels, channels, 1, 1] += 1
            conv.weight.copy_(weight)
            conv.bias.copy_(self.conv3.bias + self.conv1.bias)
        return conv


class CompactU(nn.Module):
    """Compact на половинной сетке (Э19): вход разложен pixel-unshuffle 2×2 в
    12 каналов, свёртки считают 360p вместо 720p, последний слой отдаёт 3·4²
    канала — pixelshuffle ×4 даёт сразу ×2 к входу. Умножений на пиксель
    источника — C²·9/4 на слой против 24²·9 у SUC, поле зрения вдвое шире.
    С `rep` свёртки — `RepConv` (`rep_identity` — с тождеством); `fused()` —
    та же сеть с обычными свёртками."""

    def __init__(
        self, num_feat: int = 48, num_conv: int = 7, scale: int = 2, rep: bool = False, rep_identity: bool = False
    ):
        super().__init__()
        self.scale = scale
        self.num_feat, self.num_conv = num_feat, num_conv

        def conv(cin: int, cout: int, identity: bool) -> nn.Module:
            return RepConv(cin, cout, identity and rep_identity) if rep else nn.Conv2d(cin, cout, 3, padding=1)

        layers: list[nn.Module] = [conv(12, num_feat, False), nn.PReLU(num_feat)]
        for _ in range(num_conv):
            layers += [conv(num_feat, num_feat, True), nn.PReLU(num_feat)]
        layers.append(conv(num_feat, 3 * (2 * scale) ** 2, False))
        self.body = nn.Sequential(*layers)
        # С нуля выход начинается почти с skip: иначе случайный хвост первые
        # тысячи итераций учит только погасить себя.
        with torch.no_grad():
            for tail in [layers[-1].conv3, layers[-1].conv1] if rep else [layers[-1]]:
                tail.weight.mul_(0.1)
                tail.bias.zero_()

    def fused(self) -> "CompactU":
        plain = CompactU(self.num_feat, self.num_conv, self.scale)
        plain.body = nn.Sequential(*[m.fuse() if isinstance(m, RepConv) else m for m in self.body])
        return plain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pixel_shuffle(self.body(F.pixel_unshuffle(x, 2)), 2 * self.scale)
        return out + F.interpolate(x, scale_factor=self.scale, mode="nearest")


def build(arch: str) -> nn.Module:
    """`suc` — SRVGGNetCompact 24nf/8nc; `u<C>x<N>` — CompactU с C каналами и
    N внутренними свёртками, `u<C>x<N>r` — она же с `RepConv` (3×3 + 1×1) на
    обучение, `u<C>x<N>ri` — ещё и с тождеством."""
    if arch == "suc":
        return Compact()
    base, _, rep = arch.removeprefix("u").partition("r")
    feat, conv = base.split("x")
    return CompactU(int(feat), int(conv), rep=arch != base and "r" in arch, rep_identity=rep == "i")


def deployed(model: nn.Module) -> nn.Module:
    """Сеть, как она пойдёт в ядро: у `RepConv` ветки слиты."""
    return model.fused() if isinstance(model, CompactU) and any(isinstance(m, RepConv) for m in model.body) else model


def onnx_compact(path: Path) -> Compact:
    """SRVGGNetCompact любого размера из ONNX (SUC, Compact и т. п.): число
    каналов и свёрток — по графу."""
    graph = onnx.load(str(path)).graph
    init = {t.name: t for t in graph.initializer}
    convs = [n for n in graph.node if n.op_type == "Conv"]
    feat = init[convs[0].input[1]].dims[0]
    model = Compact(feat, len(convs) - 2)
    load_onnx(model, path)
    return model


def load_any(path: Path) -> nn.Module:
    """ONNX — веса SUC; `.pt` — чекпойнт прогона, архитектура из `arch.txt`
    рядом (нет файла — SUC)."""
    if path.suffix != ".pt":
        model = Compact()
        load_onnx(model, path)
        return model.eval()
    arch_file = path.parent / "arch.txt"
    model = build(arch_file.read_text().strip() if arch_file.exists() else "suc")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def load_onnx(model: Compact, path: Path) -> None:
    """Свёртки и PReLU из графа — в порядке графа, как у `compact-to-wgsl.py`."""
    graph = onnx.load(str(path)).graph
    init = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in graph.initializer}
    convs = [n for n in graph.node if n.op_type == "Conv"]
    prelus = [n for n in graph.node if n.op_type == "PRelu"]
    mods_conv = [m for m in model.body if isinstance(m, nn.Conv2d)]
    mods_prelu = [m for m in model.body if isinstance(m, nn.PReLU)]
    if len(convs) != len(mods_conv) or len(prelus) != len(mods_prelu):
        raise SystemExit(f"в ONNX {len(convs)} свёрток и {len(prelus)} PReLU — форма не та")
    with torch.no_grad():
        for node, mod in zip(convs, mods_conv):
            mod.weight.copy_(torch.from_numpy(init[node.input[1]]))
            mod.bias.copy_(torch.from_numpy(init[node.input[2]]))
        for node, mod in zip(prelus, mods_prelu):
            mod.weight.copy_(torch.from_numpy(init[node.input[1]].reshape(-1)))


def to_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 BGR (…, H, W, 3) → float RGB (…, 3, H, W) в 0..1."""
    t = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1]))
    return t.to(device).permute(*range(t.dim() - 3), -1, -3, -2).float() / 255


def sobel(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device)
    k = torch.stack([k, k.T])[:, None].repeat(3, 1, 1, 1)
    return F.conv2d(x, k, padding=1, groups=3)


def ink_weight(target: torch.Tensor, ink: float) -> torch.Tensor:
    """Вес пикселя в L1: 1 + `ink` на штрихе цели — там, где она темнее своего
    окружения (гаусс σ = 4 px) на 20 уровней и больше (плавно с 10). Без него
    сеть на неуверенном входе рисует линии и буквы светлее цели."""
    r, g, b = target.unbind(-3)
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b)[:, None]
    x = torch.arange(-12, 13, device=target.device, dtype=target.dtype)
    k = torch.exp(-(x**2) / (2 * 4.0**2))
    k = k / k.sum()
    blur = F.conv2d(F.pad(y, (12, 12, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    blur = F.conv2d(F.pad(blur, (0, 0, 12, 12), mode="replicate"), k.view(1, 1, -1, 1))
    depth = (blur - y) * 255
    return 1 + ink * ((depth - 10) / 10).clamp(0, 1)


def gauss(y: torch.Tensor, sigma: float) -> torch.Tensor:
    r = math.ceil(3 * sigma)
    x = torch.arange(-r, r + 1, device=y.device, dtype=y.dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = (k / k.sum()).view(1, 1, 1, -1)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="replicate"), k)
    return F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.transpose(-1, -2))


def band_energy(x: torch.Tensor) -> torch.Tensor:
    """Сколько фактуры в участке: СКО полосы σ0.75–3 по яркости в окнах 16×16
    с шагом 8. Сравнение этих чисел, а не пикселей, не требует попасть в фазу
    фактуры — L1 по пикселям на неуверенном входе даёт среднее, «акварель»."""
    r, g, b = x.unbind(-3)
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b)[:, None]
    band = gauss(y, 0.75) - gauss(y, 3.0)
    return (F.avg_pool2d(band**2, 16, stride=8) + 1e-8).sqrt()


def wl1(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    return F.l1_loss(a, b) if w is None else ((a - b).abs() * w).sum() / (w.sum() * a.shape[-3])


def clean_input(gt: torch.Tensor) -> torch.Tensor:
    """Вход без порчи: эталон, уменьшенный усреднением 2×2 — то же, что
    INTER_AREA в `random_degrade` до порчи."""
    return F.avg_pool2d(gt, 2)


def area_matrix(n_in: int, n_out: int, device: torch.device) -> torch.Tensor:
    """(n_out, n_in): усреднение по площади при уменьшении n_in → n_out, в том
    числе дробном (4 → 3 у выхода ×2 против эталона 1.5×, Э28). Матрицей, а не
    `adaptive_avg_pool2d`: та на MPS берёт только кратные размеры."""
    edges = torch.arange(n_out + 1, dtype=torch.float64) * n_in / n_out
    lo, hi = edges[:-1, None], edges[1:, None]
    x = torch.arange(n_in, dtype=torch.float64)[None]
    overlap = (torch.minimum(hi, x + 1) - torch.maximum(lo, x)).clamp(min=0)
    return (overlap / overlap.sum(1, keepdim=True)).float().to(device)


def area_down(x: torch.Tensor, size: int) -> torch.Tensor:
    """Квадратный (…, C, H, H) → (…, C, size, size) усреднением по площади."""
    a = area_matrix(x.shape[-1], size, x.device)
    return a @ x @ a.T


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return 10 * math.log10(1 / F.mse_loss(a[..., 8:-8, 8:-8], b[..., 8:-8, 8:-8]).item())


def validate(model: Compact, teacher: Compact, device: torch.device) -> tuple[float, float]:
    """PSNR против эталона и против учителя на чистом входе."""
    model.eval()
    vs_gt, vs_teacher = [], []
    with torch.no_grad():
        for lq_path in sorted((DATA / "val" / "lq").glob("*.png")):
            lq = to_tensor(cv2.imread(str(lq_path)), device)[None]
            gt = to_tensor(cv2.imread(str(DATA / "val" / "gt" / lq_path.name)), device)[None]
            # В плеере выход уходит в rgba8unorm — там он и обрезается до 0..1.
            out = model(lq).clamp(0, 1)
            vs_gt.append(psnr(out, gt))
            vs_teacher.append(psnr(out, teacher(clean_input(gt)).clamp(0, 1)))
    model.train()
    return float(np.mean(vs_gt)), float(np.mean(vs_teacher))


def validate_text(model: nn.Module, text: dict[str, np.ndarray], device: torch.device) -> float:
    """PSNR на надписях (Э22): пары с проверочными шрифтами, только по маске."""
    model.eval()
    se, weight = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(text["val_lq"]), 16):
            out = model(to_tensor(text["val_lq"][i : i + 16], device)).clamp(0, 1)
            gt = to_tensor(text["val_gt"][i : i + 16], device)
            m = torch.from_numpy(text["val_mask"][i : i + 16]).to(device).float()[:, None] / 255
            se += (((out - gt) ** 2) * m).sum().item()
            weight += m.sum().item() * 3
    model.train()
    return 10 * math.log10(weight / se)


def validate_al(model: nn.Module, al: dict[str, np.ndarray], device: torch.device) -> float:
    """PSNR на проверочных сериях AniLibria (Э28): выход ×2, уменьшенный до 1.5×,
    против кадра 1080p."""
    model.eval()
    se, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(al["val_lq"]), 16):
            gt = to_tensor(al["val_gt"][i : i + 16], device)
            out = area_down(model(to_tensor(al["val_lq"][i : i + 16], device)).clamp(0, 1), gt.shape[-1])
            se += F.mse_loss(out[..., 8:-8, 8:-8], gt[..., 8:-8, 8:-8], reduction="sum").item()
            n += gt[..., 8:-8, 8:-8].numel()
    model.train()
    return 10 * math.log10(n / se)


def export(model: Compact, path: Path) -> None:
    model = model.to("cpu").eval()
    torch.onnx.export(
        model,
        torch.rand(1, 3, 64, 64),
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h2", 3: "w2"}},
        opset_version=13,
        dynamo=False,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init", type=Path, help="ONNX SUC, с которого начать; без него — с нуля")
    p.add_argument("--arch", default="suc", help="suc, u<C>x<N> (CompactU) или u<C>x<N>r (с RepConv)")
    p.add_argument(
        "--teacher-model",
        type=Path,
        default=DATA / "init" / "2x_AnimeJaNai_V2_SuperUltraCompact_100k.onnx",
        help="ONNX SRVGGNetCompact учителя любого размера (по умолчанию — AnimeJaNai V2 SUC)",
    )
    p.add_argument("--data", default="train", help="каталог пар в data/ (`make_pairs.py --out`)")
    p.add_argument(
        "--teacher-cache",
        help="npy в каталоге пар с выходами учителя на чистом входе (`teacher_cache.py`): "
        "тяжёлый учитель не считается на каждом шаге",
    )
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--iters", type=int, default=30_000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--edge", type=float, default=0.5, help="вес L1 по градиентам Собеля")
    p.add_argument(
        "--teacher",
        type=float,
        default=0.0,
        help="доля цели «исходная модель на чистом входе» вместо эталона: "
        "L1 к эталону тянет к мылу, учитель сохраняет вид JaNai",
    )
    p.add_argument(
        "--ink",
        type=float,
        default=0.0,
        help="добавочный вес L1 на штрихе цели (линии, буквы): против их высветления",
    )
    p.add_argument(
        "--text",
        help="каталог пар с надписями в data/ (`text_pairs.py`): своя цель — чёткий эталон на тексте; "
        "нужен --teacher-cache",
    )
    p.add_argument("--text-share", type=float, default=0.2, help="доля батча из пар с надписями")
    p.add_argument(
        "--text-weight",
        type=float,
        default=1.0,
        help="вес L1 на маске надписи: без него надписи — ~2%% пикселей батча, и сеть их не замечает (Э22)",
    )
    p.add_argument(
        "--texture",
        type=float,
        default=0.0,
        help="вес L1 между энергиями полосы σ0.75–3 по участкам выхода и эталона (band_energy): против «акварели»",
    )
    p.add_argument(
        "--al",
        help="каталог фрагментов AniLibria в data/ (`anilibria_pairs.py --pack`, Э28): цель — кадр 1080p, "
        "с ним сравнивается выход, уменьшенный до 1.5× входа",
    )
    p.add_argument("--al-share", type=float, default=0.25, help="доля батча из пар AniLibria")
    p.add_argument("--al-weight", type=float, default=1.0, help="вес потерь на парах AniLibria")
    p.add_argument("--from", dest="start", type=Path, help="начать с чекпойнта .pt вместо --init")
    p.add_argument("--every", type=int, default=2_000)
    p.add_argument("--mmap", action="store_true", help="данные через memmap, не в память")
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args.run.mkdir(parents=True, exist_ok=True)
    log = open(args.run / "log.txt", "a")

    def say(text: str) -> None:
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    torch.manual_seed(0)
    model = build(args.arch)
    if args.init:
        load_onnx(model, args.init)
    model = model.to(device)
    if args.start:
        model.load_state_dict(torch.load(args.start, map_location=device))
    # Чекпойнты — уже слитые (`deployed`): их берут to_wgsl.py, compare.py и
    # pilot_eval.py; неслитое состояние для продолжения — `last_train.pt`.
    plain_arch = args.arch.partition("r")[0]
    (args.run / "arch.txt").write_text(plain_arch + "\n")
    teacher = onnx_compact(args.teacher_model).to(device).eval().requires_grad_(False)
    origin = args.start or args.init
    params = sum(p.numel() for p in model.parameters())
    say(f"{args.arch} ({params} параметров), старт с {origin.name if origin else 'нуля'}, "
        f"учитель {args.teacher}, штрих {args.ink}, lr {args.lr}")
    say("val PSNR к эталону {:.3f} дБ, к учителю {:.3f} дБ".format(*validate(model, teacher, device)))

    # Целиком в память (~3 ГБ): случайное чтение из memmap держало 4 it/s.
    # С `--mmap` — memmap: параллельные прогоны делят кэш страниц, а не
    # держат по копии.
    mode = "r" if args.mmap else None
    lq_all = np.load(DATA / args.data / "lq.npy", mmap_mode=mode)
    gt_all = np.load(DATA / args.data / "gt.npy", mmap_mode=mode)
    cached = np.load(DATA / args.data / args.teacher_cache, mmap_mode=mode) if args.teacher_cache else None
    say(f"пар: {len(lq_all)} ({args.data}), учитель {args.teacher_model.name}"
        f"{', кэш ' + args.teacher_cache if cached is not None else ''}, устройство: {device}")
    text = None
    n_text = 0
    if args.text:
        if cached is None:
            raise SystemExit("--text: цель снаружи надписей — из кэша, нужен --teacher-cache")
        text = {k: np.load(DATA / args.text / f"{k}.npy")
                for k in ("lq", "gt", "target", "mask", "val_lq", "val_gt", "val_mask")}
        n_text = round(args.batch * args.text_share)
        say(f"пар с надписями: {len(text['lq'])} ({args.text}), в батче {n_text} из {args.batch}; "
            "текст в логе — PSNR по маске надписи на проверочных шрифтах")
        say(f"текст {validate_text(model, text, device):.3f} дБ")
    al = None
    n_al = 0
    if args.al:
        al = {k: np.load(DATA / args.al / part / f"{name}.npy")
              for k, part, name in (("lq", "train", "lq"), ("gt", "train", "gt"),
                                    ("val_lq", "val", "lq"), ("val_gt", "val", "gt"))}
        if al["lq"].shape[1] != lq_all.shape[1]:
            raise SystemExit(f"--al: вход {al['lq'].shape[1]} px, а у {args.data} — {lq_all.shape[1]}")
        n_al = round(args.batch * args.al_share)
        say(f"пар AniLibria: {len(al['lq'])} ({args.al}, эталон {al['gt'].shape[1]} px), "
            f"в батче {n_al} из {args.batch}, вес {args.al_weight}")
        say(f"AniLibria {validate_al(model, al, device):.3f} дБ")
    # Свой генератор: без --al поток `rng` тот же, что раньше, и старые прогоны
    # воспроизводятся.
    al_rng = np.random.default_rng(1)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=args.lr / 100)
    rng = np.random.default_rng(0)
    best = -1.0
    started = time.time()
    running = 0.0
    texture_running = 0.0
    al_running = 0.0

    for it in range(1, args.iters + 1):
        idx = np.sort(rng.choice(len(lq_all), args.batch - n_text - n_al, replace=False))
        lq = to_tensor(lq_all[idx], device)
        gt = to_tensor(gt_all[idx], device)
        pre = to_tensor(cached[idx], device) if cached is not None else None
        boost = None
        if text is not None:
            tidx = rng.choice(len(text["lq"]), n_text, replace=False)
            lq = torch.cat([lq, to_tensor(text["lq"][tidx], device)])
            gt = torch.cat([gt, to_tensor(text["gt"][tidx], device)])
            pre = torch.cat([pre, to_tensor(text["target"][tidx], device)])
            mask = torch.from_numpy(text["mask"][tidx]).to(device).float()[:, None] / 255
            boost = torch.cat([torch.ones_like(mask[:1]).expand(len(idx), -1, -1, -1),
                               1 + (args.text_weight - 1) * mask])
        # Повороты и отражения: порча x264 от них не меняется по смыслу.
        for flip in (lambda t: t.flip(-1), lambda t: t.flip(-2), lambda t: t.transpose(-1, -2)):
            if rng.random() < 0.5:
                lq, gt = flip(lq), flip(gt)
                pre = flip(pre) if pre is not None else None
                boost = flip(boost) if boost is not None else None
        if al is not None:
            aidx = np.sort(al_rng.choice(len(al["lq"]), n_al, replace=False))
            al_lq = to_tensor(al["lq"][aidx], device)
            al_gt = to_tensor(al["gt"][aidx], device)
            for flip in (lambda t: t.flip(-1), lambda t: t.flip(-2), lambda t: t.transpose(-1, -2)):
                if al_rng.random() < 0.5:
                    al_lq, al_gt = flip(al_lq), flip(al_gt)
            out_all = model(torch.cat([lq, al_lq]))
            out, al_out = out_all[: len(lq)], area_down(out_all[len(lq) :], al_gt.shape[-1])
        else:
            out = model(lq)
        loss = 0.0
        if args.teacher < 1:
            w = ink_weight(gt, args.ink) if args.ink else None
            loss += (1 - args.teacher) * (wl1(out, gt, w) + args.edge * wl1(sobel(out), sobel(gt), w))
        if args.teacher > 0:
            with torch.no_grad():
                target = pre if pre is not None else teacher(clean_input(gt))
                w = ink_weight(target, args.ink) if args.ink else None
                if boost is not None:
                    w = boost if w is None else w * boost
            loss += args.teacher * (wl1(out, target, w) + args.edge * wl1(sobel(out), sobel(target), w))
        if args.texture:
            tex = F.l1_loss(band_energy(out), band_energy(gt))
            loss = loss + args.texture * tex
            texture_running += tex.item()
        if al is not None:
            # Средние по своим частям батча, сложенные в долях батча: на старой
            # части те же потери, что без --al.
            w = ink_weight(al_gt, args.ink) if args.ink else None
            al_loss = wl1(al_out, al_gt, w) + args.edge * wl1(sobel(al_out), sobel(al_gt), w)
            share = n_al / args.batch
            loss = (1 - share) * loss + share * args.al_weight * al_loss
            al_running += al_loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()

        if it % 200 == 0:
            rate = it / (time.time() - started)
            tail = f"  фактура {texture_running / 200:.5f}" if args.texture else ""
            tail += f"  AniLibria {al_running / 200:.5f}" if al is not None else ""
            say(f"{it:>6}  loss {running / 200:.5f}{tail}  lr {sched.get_last_lr()[0]:.2e}  {rate:.1f} it/s")
            running = texture_running = al_running = 0.0
        if it % args.every == 0 or it == args.iters:
            vs_gt, vs_teacher = validate(model, teacher, device)
            # «Лучший» — по той цели, к которой модель и тянется.
            value = vs_teacher * args.teacher + vs_gt * (1 - args.teacher)
            state = deployed(model).state_dict()
            torch.save(state, args.run / "last.pt")
            torch.save(state, args.run / f"it{it}.pt")
            torch.save(model.state_dict(), args.run / "last_train.pt")
            mark = ""
            if value > best:
                best = value
                torch.save(state, args.run / "best.pt")
                mark = "  ← лучший"
            tail = f", текст {validate_text(model, text, device):.3f} дБ" if text is not None else ""
            tail += f", AniLibria {validate_al(model, al, device):.3f} дБ" if al is not None else ""
            say(f"{it:>6}  val PSNR к эталону {vs_gt:.3f} дБ, к учителю {vs_teacher:.3f} дБ{tail}{mark}")

    model = build(plain_arch)
    model.load_state_dict(torch.load(args.run / "best.pt"))
    export(model, args.run / "model.onnx")
    say(f"лучшая цель {best:.3f} дБ, экспорт: {args.run / 'model.onnx'}")


if __name__ == "__main__":
    main()
