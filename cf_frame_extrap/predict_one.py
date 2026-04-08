from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from cf_frame_extrap.data import ActionNorm
from cf_frame_extrap.model import ActionConditionedUNet, ModelConfig
from cf_frame_extrap.utils import ensure_dir


def _read_gray_01(path: Path, height: int, width: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def _load_ckpt(path: str) -> dict:
    # PyTorch >=2.6 defaults weights_only=True which can reject our training checkpoints.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True, help="Output PNG path.")

    p.add_argument("--height", type=int, default=244)
    p.add_argument("--width", type=int, default=324)

    p.add_argument("--img", type=str, nargs=3, required=True, help="3 images: oldest -> newest.")
    p.add_argument(
        "--action",
        type=float,
        nargs=12,
        required=True,
        metavar=("C0", "T0", "B0", "R0", "C1", "T1", "B1", "R1", "C2", "T2", "B2", "R2"),
        help="3 CTBR actions (oldest -> newest), total 12 floats.",
    )
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    ckpt = _load_ckpt(args.ckpt)

    model_cfg = ckpt.get("model_cfg") or {"k": 3}
    cfg = ModelConfig(**model_cfg)
    cfg.k = 3  # this script is explicitly for 3-frame input

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionConditionedUNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    an = ckpt.get("action_norm", None)
    if an is None:
        raise ValueError("Checkpoint missing action_norm; train with cf_frame_extrap.train.")
    action_norm = ActionNorm(mean=np.array(an["mean"], dtype=np.float32), std=np.array(an["std"], dtype=np.float32))

    frames = np.stack([_read_gray_01(Path(p), args.height, args.width) for p in args.img], axis=0).astype(np.float32)
    a = np.array(args.action, dtype=np.float32).reshape(3, 4)
    a = action_norm.normalize(a).astype(np.float32)

    frames_t = torch.from_numpy(frames[None, ...]).to(device)  # [1,3,H,W]
    actions_t = torch.from_numpy(a[None, ...]).to(device)  # [1,3,4]

    pred = model(frames_t, actions_t)[0, 0].detach().cpu().numpy()  # [H,W] in [0,1]
    pred_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), pred_u8)


if __name__ == "__main__":
    main()

