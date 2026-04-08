from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cf_frame_extrap.data import ActionNorm, FrameSequenceDataset, discover_episodes, split_episodes
from cf_frame_extrap.losses import recon_loss
from cf_frame_extrap.model import ActionConditionedUNet, ModelConfig
from cf_frame_extrap.utils import AverageMeter, count_parameters, ensure_dir, psnr, save_checkpoint, set_seed, to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--k", type=int, default=3)
    p.add_argument("--height", type=int, default=244)
    p.add_argument("--width", type=int, default=324)
    p.add_argument("--hist_stride", type=int, default=6, help="History spacing S (use frames t-2S, t-S, t).")
    p.add_argument("--pred_horizon", type=int, default=2, help="Predict target at t+H.")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.1)

    p.add_argument("--base_ch", type=int, default=48)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--w_l1", type=float, default=1.0)
    p.add_argument("--w_ssim", type=float, default=0.5)
    p.add_argument("--w_gdl", type=float, default=0.2)

    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model: ActionConditionedUNet,
    loader: DataLoader,
    device: torch.device,
    loss_weights: Tuple[float, float, float],
) -> Dict[str, float]:
    model.eval()
    w_l1, w_ssim, w_gdl = loss_weights

    loss_m = AverageMeter()
    psnr_m = AverageMeter()

    for batch in loader:
        batch = to_device(batch, device)
        pred = model(batch["frames"], batch["actions"])
        loss = recon_loss(pred, batch["target"], w_l1=w_l1, w_ssim=w_ssim, w_gdl=w_gdl)
        loss_m = loss_m.update(loss.item(), n=pred.shape[0])
        psnr_m = psnr_m.update(psnr(pred, batch["target"]).mean().item(), n=pred.shape[0])

    return {"loss": loss_m.avg, "psnr": psnr_m.avg}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = ensure_dir(args.out_dir)
    (out_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episodes = discover_episodes(args.data_root)
    train_eps, val_eps = split_episodes(episodes, val_ratio=args.val_ratio, seed=args.seed)

    # Compute action normalization ONLY on training episodes, store for val+checkpoint
    action_norm = FrameSequenceDataset(
        train_eps,
        k=args.k,
        height=args.height,
        width=args.width,
        train=True,
        hist_stride=args.hist_stride,
        pred_horizon=args.pred_horizon,
    ).action_norm

    ds_train = FrameSequenceDataset(
        train_eps,
        k=args.k,
        height=args.height,
        width=args.width,
        train=True,
        action_norm=action_norm,
        hist_stride=args.hist_stride,
        pred_horizon=args.pred_horizon,
    )
    ds_val = FrameSequenceDataset(
        val_eps,
        k=args.k,
        height=args.height,
        width=args.width,
        train=False,
        action_norm=action_norm,
        hist_stride=args.hist_stride,
        pred_horizon=args.pred_horizon,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    cfg = ModelConfig(k=args.k, base_ch=args.base_ch, dropout=args.dropout)
    model = ActionConditionedUNet(cfg).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    loss_weights = (args.w_l1, args.w_ssim, args.w_gdl)

    meta = {
        "args": vars(args),
        "model_cfg": asdict(cfg),
        "n_params": count_parameters(model),
        "device": str(device),
        "action_norm": {"mean": action_norm.mean.tolist(), "std": action_norm.std.tolist()},
        "n_episodes": len(episodes),
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "n_train_samples": len(ds_train),
        "n_val_samples": len(ds_val),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_m = AverageMeter()
        psnr_m = AverageMeter()

        pbar = tqdm(dl_train, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            batch = to_device(batch, device)
            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                pred = model(batch["frames"], batch["actions"])
                loss = recon_loss(pred, batch["target"], *loss_weights)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()

            bsz = pred.shape[0]
            loss_m = loss_m.update(loss.item(), n=bsz)
            psnr_m = psnr_m.update(psnr(pred.detach(), batch["target"]).mean().item(), n=bsz)

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1
            pbar.set_postfix(loss=f"{loss_m.avg:.4f}", psnr=f"{psnr_m.avg:.2f}")

        val_metrics = evaluate(model, dl_val, device, loss_weights)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/psnr", val_metrics["psnr"], epoch)

        ckpt_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model_cfg": asdict(cfg),
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "action_norm": {"mean": action_norm.mean, "std": action_norm.std},
            "val_metrics": val_metrics,
            "meta": meta,
        }
        save_checkpoint(out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", ckpt_payload)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(out_dir / "best.pt", ckpt_payload)

        # also keep a "last" pointer
        save_checkpoint(out_dir / "last.pt", ckpt_payload)

    writer.close()


if __name__ == "__main__":
    main()

