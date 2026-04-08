from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from cf_frame_extrap.data import ActionNorm, EpisodeIndex, discover_episodes
from cf_frame_extrap.losses import ssim
from cf_frame_extrap.model import ActionConditionedUNet, ModelConfig
from cf_frame_extrap.utils import AverageMeter, ensure_dir, psnr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--k", type=int, default=3)
    p.add_argument("--height", type=int, default=244)
    p.add_argument("--width", type=int, default=324)

    p.add_argument("--save_gt", action="store_true", help="Also save ground truth frames for easy comparison.")
    p.add_argument("--max_frames_per_episode", type=int, default=-1, help="Limit (after warmup). -1 = all.")
    return p.parse_args()


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


@torch.no_grad()
def infer_episode(
    model: ActionConditionedUNet,
    ep: EpisodeIndex,
    action_norm: ActionNorm,
    k: int,
    height: int,
    width: int,
    out_dir: Path,
    save_gt: bool,
    max_frames: int,
    device: torch.device,
) -> Dict[str, float]:
    pred_dir = ensure_dir(out_dir / ep.episode_dir.name / "pred")
    gt_dir = ensure_dir(out_dir / ep.episode_dir.name / "gt") if save_gt else None

    loss_psnr = AverageMeter()
    loss_ssim = AverageMeter()

    # Teacher forcing: history comes from real frames, not predictions.
    t_end = ep.T
    if max_frames is not None and max_frames > 0:
        t_end = min(t_end, k + max_frames)

    for t in tqdm(range(k, t_end), desc=ep.episode_dir.name, leave=False):
        frames_hist = np.stack(
            [_read_gray_01(p, height, width) for p in ep.frame_paths[t - k : t]],
            axis=0,
        )  # [K,H,W]
        gt = _read_gray_01(ep.frame_paths[t], height, width)  # [H,W]

        a_hist = ep.actions[t - k : t]
        a_hist = action_norm.normalize(a_hist).astype(np.float32)

        frames_t = torch.from_numpy(frames_hist[None, ...]).to(device)
        actions_t = torch.from_numpy(a_hist[None, ...]).to(device)

        pred = model(frames_t, actions_t)[0, 0]  # [H,W]
        gt_t = torch.from_numpy(gt[None, None, ...]).to(device)  # [1,1,H,W]
        pred_t = pred[None, None, ...]

        loss_psnr = loss_psnr.update(psnr(pred_t, gt_t).mean().item(), n=1)
        loss_ssim = loss_ssim.update(ssim(pred_t, gt_t).mean().item(), n=1)

        pred_u8 = (pred.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        cv2.imwrite(str(pred_dir / f"{t:06d}.png"), pred_u8)
        if gt_dir is not None:
            gt_u8 = (np.clip(gt, 0, 1) * 255.0).astype(np.uint8)
            cv2.imwrite(str(gt_dir / f"{t:06d}.png"), gt_u8)

    return {"psnr": loss_psnr.avg, "ssim": loss_ssim.avg, "n": int(max(0, t_end - k))}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

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
    cfg.k = args.k

    model = ActionConditionedUNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    an = ckpt.get("action_norm", None)
    if an is None:
        raise ValueError("Checkpoint missing action_norm; train with cf_frame_extrap.train.")
    action_norm = ActionNorm(mean=np.array(an["mean"], dtype=np.float32), std=np.array(an["std"], dtype=np.float32))

    episodes = discover_episodes(args.data_root)

    metrics: Dict[str, Dict[str, float]] = {}
    psnr_m = AverageMeter()
    ssim_m = AverageMeter()
    n_m = 0

    for ep in episodes:
        m = infer_episode(
            model=model,
            ep=ep,
            action_norm=action_norm,
            k=args.k,
            height=args.height,
            width=args.width,
            out_dir=out_dir,
            save_gt=bool(args.save_gt),
            max_frames=args.max_frames_per_episode,
            device=device,
        )
        metrics[ep.episode_dir.name] = m
        if m["n"] > 0:
            psnr_m = psnr_m.update(m["psnr"], n=m["n"])
            ssim_m = ssim_m.update(m["ssim"], n=m["n"])
            n_m += int(m["n"])

    summary = {"overall": {"psnr": psnr_m.avg, "ssim": ssim_m.avg, "n": n_m}, "per_episode": metrics}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

