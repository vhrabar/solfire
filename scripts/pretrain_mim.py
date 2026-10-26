"""
Masked-image pretraining of a segmentation network on unlabelled GONG H-alpha frames (D1).

SimMIM-style: random 32 px patches of a native-resolution crop are hidden (filled with the crop's
mean) and the whole encoder-decoder learns to reconstruct them. The target is normalised per patch,
so the loss rewards the dark filament and plage structure rather than the limb darkening. The
saved checkpoint holds the encoder and decoder without the output head and starts a supervised run
through ``--init-checkpoint``.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.networks.pretrained import Pretrained  # noqa: E402
from src.training.engine import atomic_save  # noqa: E402

GONG = ROOT / "runs" / "external" / "gong"


def usable_frames(gong_dir) -> list[Path]:
    """
    The H-alpha frames that belong to a burst which passed quality control.
    :param gong_dir: Directory holding ``manifest.json`` and ``jpeg/``
    :return: Sorted JPEG paths
    """
    manifest = json.loads((Path(gong_dir) / "manifest.json").read_text())
    names = {frame for burst in manifest["bursts"] for frame in burst}
    return sorted(p for p in (Path(gong_dir) / "jpeg").glob("*.jpeg") if p.stem in names)


class MaskedCropDataset(Dataset):
    """
    Native-resolution crops centred on the solar disk, with flips and rotations.
    """

    def __init__(self, paths, crop=512, repeat=8, disk_level=0.1):
        """
        :param paths: Observation files
        :param crop: Edge length of the returned window
        :param repeat: Crops drawn per observation and epoch
        :param disk_level: Intensity a pixel needs to count as on the disk
        """
        self.paths = list(paths)
        self.crop = crop
        self.repeat = repeat
        self.disk_level = round(disk_level * 255)

    def __len__(self):
        """
        :return: Number of crops in one epoch
        """
        return len(self.paths) * self.repeat

    def __getitem__(self, index):
        """
        :param index: Position in the epoch; observations are cycled ``repeat`` times
        :return: ``(1, crop, crop)`` float tensor in [0, 1]
        """
        with Image.open(self.paths[index % len(self.paths)]) as handle:
            image = np.asarray(handle.convert("L"), dtype=np.uint8)
        size = image.shape[0]
        limit = size - self.crop
        ys, xs = np.nonzero(image[::8, ::8] >= self.disk_level)
        pick = random.randrange(len(ys))
        x = min(max(int(xs[pick]) * 8 - self.crop // 2, 0), limit)
        y = min(max(int(ys[pick]) * 8 - self.crop // 2, 0), limit)
        window = torch.from_numpy(image[y : y + self.crop, x : x + self.crop].copy()).float().div_(255.0)
        if random.random() < 0.5:
            window = window.flip(-1)
        window = torch.rot90(window, random.randint(0, 3), (-2, -1))
        return window.unsqueeze(0)


def random_patch_mask(batch, size, patch, ratio, device) -> torch.Tensor:
    """
    :param batch: Number of crops
    :param size: Crop edge length, a multiple of ``patch``
    :param patch: Edge length of one hidden square
    :param ratio: Share of squares hidden
    :param device: Device of the result
    :return: ``(B, 1, size, size)`` float mask, 1 where hidden
    """
    cells = size // patch
    scores = torch.rand(batch, cells * cells, device=device)
    hidden = scores.argsort(dim=1) < round(ratio * cells * cells)
    grid = hidden.view(batch, 1, cells, cells).float()
    return F.interpolate(grid, scale_factor=patch, mode="nearest")


def patch_normalised(image, patch) -> torch.Tensor:
    """
    Standardise every ``patch`` square on its own, as MAE does with its targets.
    :param image: ``(B, 1, H, W)`` batch
    :param patch: Edge length of one square
    :return: Batch of the same shape
    """
    mean = F.avg_pool2d(image, patch)
    var = F.avg_pool2d(image**2, patch) - mean**2
    mean = F.interpolate(mean, scale_factor=patch, mode="nearest")
    std = F.interpolate(var.clamp_min(0).add(1e-6).sqrt(), scale_factor=patch, mode="nearest")
    return (image - mean) / std


def main() -> None:
    """
    Parse arguments, pretrain, and write ``last.pt`` (resumable) and ``encoder.pt`` (for ``--init-checkpoint``).
    :return: None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decoder", default="unetplusplus")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gong", default=str(GONG))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=8, help="crops per frame and epoch")
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    paths = usable_frames(args.gong)
    dataset = MaskedCropDataset(paths, crop=args.crop, repeat=args.repeat)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )
    print(f"{len(paths)} frames, {len(dataset)} crops per epoch")

    model_config = {"in_channels": 1, "out_channels": 1, "decoder": args.decoder, "encoder": args.encoder}
    model = Pretrained(**model_config, encoder_weights="imagenet").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(loader)
    warmup = args.warmup_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))
    )

    start = 0
    last = output / "last.pt"
    if last.is_file():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["epoch"]
        print(f"resuming after epoch {start}")

    history = []
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        began, total = time.time(), 0.0
        for image in loader:
            image = image.to(device, non_blocking=True)
            hidden = random_patch_mask(image.shape[0], args.crop, args.patch, args.mask_ratio, device)
            fill = image.mean(dim=(-2, -1), keepdim=True)
            masked = image * (1 - hidden) + fill * hidden
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(masked)
            target = patch_normalised(image, args.patch)
            loss = ((prediction.float() - target).abs() * hidden).sum() / hidden.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        mean_loss = total / len(loader)
        history.append({"epoch": epoch, "loss": mean_loss, "seconds": round(time.time() - began)})
        print(f"epoch {epoch}/{args.epochs} loss {mean_loss:.4f} ({time.time() - began:.0f} s)", flush=True)
        atomic_save(
            {
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "model_config": model_config,
                "args": vars(args),
            },
            last,
        )

    # the reconstruction head is dropped so the supervised run starts its output layer fresh
    encoder_state = {k: v for k, v in model.state_dict().items() if "segmentation_head" not in k}
    atomic_save(
        {"model_state": encoder_state, "epoch": args.epochs, "model_config": model_config, "args": vars(args)},
        output / "encoder.pt",
    )
    (output / "history.json").write_text(json.dumps(history, indent=1))
    print(f"wrote {output / 'encoder.pt'}")


if __name__ == "__main__":
    main()
