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
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def _sorted_images(dir_path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [p for p in dir_path.iterdir() if p.suffix.lower() in exts]
    paths.sort(key=lambda p: p.name)
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--height", type=int, default=244)
    p.add_argument("--width", type=int, default=324)

    p.add_argument("--init_frames_dir", type=str, required=True, help="Directory with at least K initial frames.")
    p.add_argument("--actions_npy", type=str, required=True, help="(T,4) CTBR actions aligned to rollout time.")
    p.add_argument("--start_t", type=int, default=0, help="Index into actions array for first predicted frame time.")
    p.add_argument("--steps", type=int, default=300, help="How many frames to predict forward.")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    (out_dir / "pred").mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # PyTorch >=2.6 defaults weights_only=True which can reject our training checkpoints
    # (they include numpy arrays + metadata). This is safe to disable for self-generated checkpoints.
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")

    model_cfg = ckpt.get("model_cfg", None)
    if model_cfg is None:
        model_cfg = {"k": args.k}
    cfg = ModelConfig(**model_cfg)
    cfg.k = args.k  # force K from CLI for rollout buffers

    model = ActionConditionedUNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    an = ckpt.get("action_norm", None)
    if an is None:
        raise ValueError("Checkpoint missing action_norm; train with cf_frame_extrap.train.")
    action_norm = ActionNorm(mean=np.array(an["mean"], dtype=np.float32), std=np.array(an["std"], dtype=np.float32))

    init_dir = Path(args.init_frames_dir)
    init_paths = _sorted_images(init_dir)
    if len(init_paths) < args.k:
        raise ValueError(f"Need at least K={args.k} initial frames in {init_dir}, got {len(init_paths)}")

    # Initial buffer: last K frames
    frame_buf = [_read_gray_01(p, args.height, args.width) for p in init_paths[-args.k :]]  # list of [H,W]

    actions = np.load(args.actions_npy).astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError(f"actions_npy must have shape (T,4), got {actions.shape}")

    # Rollout: at each step s, predict frame at time t = start_t + s
    for s in range(args.steps):
        t = args.start_t + s
        if t < args.k:
            raise ValueError("start_t must be >= K so there is enough action history.")
        if t >= actions.shape[0]:
            break

        # Build tensors
        frames_hist = np.stack(frame_buf, axis=0)  # [K,H,W]
        a_hist = actions[t - args.k : t]  # [K,4]
        a_hist = action_norm.normalize(a_hist).astype(np.float32)

        frames_t = torch.from_numpy(frames_hist[None, ...]).to(device)  # [1,K,H,W]
        actions_t = torch.from_numpy(a_hist[None, ...]).to(device)  # [1,K,4]

        pred = model(frames_t, actions_t)[0, 0].detach().cpu().numpy()  # [H,W] in [0,1]
        pred_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / "pred" / f"{t:06d}.png"), pred_u8)

        # Slide buffer
        frame_buf = frame_buf[1:] + [pred]


if __name__ == "__main__":
    main()

