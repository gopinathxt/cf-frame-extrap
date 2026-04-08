from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from cf_frame_extrap.data import (
    ActionNorm,
    EpisodeIndex,
    RenderCaptureEpisode,
    discover_episodes,
    extract_capture_zip,
)
from cf_frame_extrap.model import ActionConditionedUNet, ModelConfig
from cf_frame_extrap.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Teacher-forced inference with spaced history [t-2S, t-S, t] and target t+H. "
            "Predictions are saved using the exact ground-truth filename."
        )
    )
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (best.pt / last.pt).")
    p.add_argument("--data_root", type=str, required=True, help="Dataset root (episode_*/frames + actions.npy).")
    p.add_argument("--out_dir", type=str, required=True, help="Output root for predictions.")
    p.add_argument("--height", type=int, default=244)
    p.add_argument("--width", type=int, default=324)
    p.add_argument("--k", type=int, default=3, help="Must be 3 for [t-2S, t-S, t].")
    p.add_argument("--hist_stride", type=int, default=6, help="S in [t-2S, t-S, t].")
    p.add_argument("--pred_horizon", type=int, default=2, help="H in target t+H.")
    p.add_argument(
        "--autoregressive",
        action="store_true",
        help=(
            "Run real-time autoregressive mode: seed with real [t-2S, t-S, t], then roll using predictions "
            "(e.g. [6,12,14(pred)] -> 16)."
        ),
    )
    p.add_argument(
        "--start_t",
        type=int,
        default=-1,
        help="Start t for autoregressive seed [t-2S,t-S,t]. Default: 2S.",
    )
    p.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=-1,
        help="Autoregressive-only step limit. -1 means run until episode end.",
    )
    p.add_argument(
        "--camera_stride",
        type=int,
        default=-1,
        help=(
            "Autoregressive-only: cadence for real camera frames. "
            "If target_t is on this cadence, ingest GT frame instead of model prediction. "
            "Default -1 means use hist_stride."
        ),
    )
    p.add_argument(
        "--allow_actions_npy",
        action="store_true",
        help="Allow fallback discovery that may use actions.npy episodes. Default behavior is CSV actions.",
    )
    p.add_argument(
        "--capture_data_dir",
        type=str,
        default="",
        help="Optional override for folder containing capture_*.csv (defaults to <data_root>/capture_data).",
    )
    p.add_argument("--copy_gt", action="store_true", help="Also copy resized ground-truth targets for side-by-side.")
    p.add_argument("--max_frames_per_episode", type=int, default=-1, help="Limit predictions per episode.")
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


def _load_ckpt(path: str) -> dict:
    # PyTorch >=2.6 defaults weights_only=True, which can reject checkpoints with metadata.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_compare_artifacts(pred_01: np.ndarray, gt_01: np.ndarray, out_name: str, cmp_dir: Path, mse_dir: Path) -> float:
    pred_u8 = np.clip(pred_01 * 255.0, 0, 255).astype(np.uint8)
    gt_u8 = np.clip(gt_01 * 255.0, 0, 255).astype(np.uint8)

    err = (gt_01.astype(np.float32) - pred_01.astype(np.float32)) ** 2  # per-pixel squared error in [0,1]
    mse_scalar = float(err.mean())

    # Visual error map for (ground truth - model output)^2.
    err_vis = np.clip(err * 1000, 0, 255).astype(np.uint8)
    ok_mse = cv2.imwrite(str(mse_dir / out_name), err_vis)
    if not ok_mse:
        raise RuntimeError(f"Failed to write MSE map image: {mse_dir / out_name}")

    # 3-panel strip for manual inspection: pred | gt | mse_map
    side = np.concatenate([pred_u8, gt_u8, err_vis], axis=1)
    ok_side = cv2.imwrite(str(cmp_dir / out_name), side)
    if not ok_side:
        raise RuntimeError(f"Failed to write compare image: {cmp_dir / out_name}")

    return mse_scalar


def _discover_csv_episodes(data_root: Path, capture_data_dir: str) -> List[RenderCaptureEpisode]:
    renders_root = data_root / "renders"
    if not renders_root.exists():
        raise FileNotFoundError(
            f"--actions_from_csv expects render folders under {renders_root}. "
            "Expected names like renders_capture_XX."
        )

    csv_root = Path(capture_data_dir) if capture_data_dir else (data_root / "capture_data")
    csv_root.mkdir(parents=True, exist_ok=True)
    csvs = sorted(csv_root.glob("capture_*.csv"))
    if not csvs:
        capture_zip = data_root / "capture.zip"
        if capture_zip.exists():
            csvs = extract_capture_zip(capture_zip, csv_root)
        else:
            raise FileNotFoundError(
                f"No capture_*.csv found in {csv_root} and no capture.zip at {capture_zip}."
            )

    csv_by_key = {p.stem: p for p in csvs}  # capture_01 -> path
    episodes: List[RenderCaptureEpisode] = []
    for frames_dir in sorted(renders_root.glob("renders*"), key=lambda p: p.name):
        if not frames_dir.is_dir():
            continue
        key = frames_dir.name.replace("renders_", "")  # renders_capture_01 -> capture_01
        csv_path = csv_by_key.get(key)
        if csv_path is None:
            raise FileNotFoundError(f"Missing CSV for {frames_dir.name}: expected key {key} in {csv_root}")
        episodes.append(RenderCaptureEpisode(frames_dir=frames_dir, csv_path=csv_path))

    if not episodes:
        raise FileNotFoundError(f"No render episode folders found under {renders_root}")
    return episodes


@torch.no_grad()
def infer_episode(
    model: ActionConditionedUNet,
    ep: EpisodeIndex,
    action_norm: ActionNorm,
    height: int,
    width: int,
    k: int,
    hist_stride: int,
    pred_horizon: int,
    out_dir: Path,
    copy_gt: bool,
    max_frames_per_episode: int,
    device: torch.device,
) -> int:
    pred_dir = ensure_dir(out_dir / ep.episode_dir.name / "pred")
    gt_dir = ensure_dir(out_dir / ep.episode_dir.name / "gt") if copy_gt else None
    cmp_dir = ensure_dir(out_dir / ep.episode_dir.name / "compare")
    mse_dir = ensure_dir(out_dir / ep.episode_dir.name / "mse_map")
    mse_txt = out_dir / ep.episode_dir.name / "mse_values.txt"
    mse_txt.write_text("filename\tmse\n", encoding="utf-8")

    if k != 3:
        raise ValueError("This script expects k=3 for spaced history [t-2S, t-S, t].")
    if hist_stride < 1:
        raise ValueError("hist_stride must be >= 1.")
    if pred_horizon < 1:
        raise ValueError("pred_horizon must be >= 1.")

    t_min = 2 * hist_stride
    t_max = ep.T - pred_horizon - 1
    if t_max < t_min:
        return 0

    ts: List[int] = list(range(t_min, t_max + 1))
    if max_frames_per_episode > 0:
        ts = ts[:max_frames_per_episode]

    written = 0
    for t in tqdm(ts, desc=ep.episode_dir.name, leave=False):
        hist_paths = [ep.frame_paths[t - 2 * hist_stride], ep.frame_paths[t - hist_stride], ep.frame_paths[t]]
        target_path = ep.frame_paths[t + pred_horizon]

        # Guard against data leakage: target frame must not be part of inference inputs.
        if any(p.resolve() == target_path.resolve() for p in hist_paths):
            raise RuntimeError(
                f"Target leakage detected in {ep.episode_dir.name}: target={target_path.name} "
                f"appears in history {[p.name for p in hist_paths]}"
            )

        frames_hist = np.stack([_read_gray_01(p, height, width) for p in hist_paths], axis=0).astype(np.float32)
        a_hist = np.stack(
            [ep.actions[t - 2 * hist_stride], ep.actions[t - hist_stride], ep.actions[t]],
            axis=0,
        )
        a_hist = action_norm.normalize(a_hist).astype(np.float32)

        frames_t = torch.from_numpy(frames_hist[None, ...]).to(device)
        actions_t = torch.from_numpy(a_hist[None, ...]).to(device)
        pred = model(frames_t, actions_t)[0, 0].detach().cpu().numpy()

        pred_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
        out_name = target_path.name
        ok = cv2.imwrite(str(pred_dir / out_name), pred_u8)
        if not ok:
            raise RuntimeError(f"Failed to write prediction: {pred_dir / out_name}")

        if gt_dir is not None:
            gt = _read_gray_01(target_path, height, width)
            gt_u8 = np.clip(gt * 255.0, 0, 255).astype(np.uint8)
            ok_gt = cv2.imwrite(str(gt_dir / out_name), gt_u8)
            if not ok_gt:
                raise RuntimeError(f"Failed to write GT: {gt_dir / out_name}")
        else:
            gt = _read_gray_01(target_path, height, width)

        mse_scalar = _save_compare_artifacts(pred, gt, out_name, cmp_dir=cmp_dir, mse_dir=mse_dir)
        with mse_txt.open("a", encoding="utf-8") as f:
            f.write(f"{out_name}\t{mse_scalar:.8f}\n")

        written += 1

    return written


@torch.no_grad()
def infer_episode_autoregressive(
    model: ActionConditionedUNet,
    ep: EpisodeIndex,
    action_norm: ActionNorm,
    height: int,
    width: int,
    k: int,
    hist_stride: int,
    pred_horizon: int,
    out_dir: Path,
    copy_gt: bool,
    start_t: int,
    max_steps_per_episode: int,
    camera_stride: int,
    device: torch.device,
) -> int:
    pred_dir = ensure_dir(out_dir / ep.episode_dir.name / "pred")
    gt_dir = ensure_dir(out_dir / ep.episode_dir.name / "gt") if copy_gt else None
    cmp_dir = ensure_dir(out_dir / ep.episode_dir.name / "compare")
    mse_dir = ensure_dir(out_dir / ep.episode_dir.name / "mse_map")
    mse_txt = out_dir / ep.episode_dir.name / "mse_values.txt"
    mse_txt.write_text("filename\tmse\n", encoding="utf-8")

    if k != 3:
        raise ValueError("Autoregressive mode expects k=3.")
    if hist_stride < 1:
        raise ValueError("hist_stride must be >= 1.")
    if pred_horizon < 1:
        raise ValueError("pred_horizon must be >= 1.")
    cam_stride = hist_stride if camera_stride < 1 else int(camera_stride)
    if cam_stride < 1:
        raise ValueError("camera_stride must be >= 1.")

    t_seed = (2 * hist_stride) if start_t < 0 else start_t
    seed_times = [t_seed - 2 * hist_stride, t_seed - hist_stride, t_seed]
    if seed_times[0] < 0:
        raise ValueError(
            f"Invalid start_t={t_seed} for hist_stride={hist_stride}. Need t-2S >= 0."
        )
    if seed_times[-1] >= ep.T:
        return 0

    # Start from real observed frames.
    frame_hist = [_read_gray_01(ep.frame_paths[t], height, width) for t in seed_times]
    time_hist = list(seed_times)

    steps_done = 0
    pbar = tqdm(desc=ep.episode_dir.name, leave=False)
    while True:
        t_cur = time_hist[-1]
        target_t = t_cur + pred_horizon
        if target_t >= ep.T:
            break
        if max_steps_per_episode > 0 and steps_done >= max_steps_per_episode:
            break

        # Guard against leakage in autoregressive mode as well.
        if target_t in time_hist:
            raise RuntimeError(
                f"Target leakage detected in {ep.episode_dir.name}: target_t={target_t} in history {time_hist}"
            )

        target_path = ep.frame_paths[target_t]
        out_name = target_path.name
        gt = _read_gray_01(target_path, height, width)

        # If a real camera frame arrives at this timestamp, ingest it instead of model output.
        if target_t % cam_stride == 0:
            # No prediction is written for camera-observed timestamps.
            next_frame = gt
        else:
            frames_hist = np.stack(frame_hist, axis=0).astype(np.float32)
            a_hist = np.stack([ep.actions[t] for t in time_hist], axis=0)
            a_hist = action_norm.normalize(a_hist).astype(np.float32)

            frames_t = torch.from_numpy(frames_hist[None, ...]).to(device)
            actions_t = torch.from_numpy(a_hist[None, ...]).to(device)
            pred = model(frames_t, actions_t)[0, 0].detach().cpu().numpy()

            pred_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
            ok = cv2.imwrite(str(pred_dir / out_name), pred_u8)
            if not ok:
                raise RuntimeError(f"Failed to write prediction: {pred_dir / out_name}")
            next_frame = pred.astype(np.float32)
            mse_scalar = _save_compare_artifacts(next_frame, gt, out_name, cmp_dir=cmp_dir, mse_dir=mse_dir)
            with mse_txt.open("a", encoding="utf-8") as f:
                f.write(f"{out_name}\t{mse_scalar:.8f}\n")

        if gt_dir is not None:
            gt_u8 = np.clip(gt * 255.0, 0, 255).astype(np.uint8)
            ok_gt = cv2.imwrite(str(gt_dir / out_name), gt_u8)
            if not ok_gt:
                raise RuntimeError(f"Failed to write GT: {gt_dir / out_name}")

        # Slide with the frame actually available at target_t (real if observed, else predicted).
        frame_hist = [frame_hist[1], frame_hist[2], next_frame]
        time_hist = [time_hist[1], time_hist[2], target_t]

        steps_done += 1
        pbar.update(1)

    pbar.close()
    return steps_done


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    ckpt = _load_ckpt(args.ckpt)
    model_cfg = ckpt.get("model_cfg") or {"k": args.k}
    cfg = ModelConfig(**model_cfg)
    cfg.k = args.k

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionConditionedUNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    an = ckpt.get("action_norm", None)
    if an is None:
        raise ValueError("Checkpoint missing action_norm; train with cf_frame_extrap.train.")
    action_norm = ActionNorm(mean=np.array(an["mean"], dtype=np.float32), std=np.array(an["std"], dtype=np.float32))

    if args.allow_actions_npy:
        episodes = discover_episodes(args.data_root)
    else:
        episodes = _discover_csv_episodes(Path(args.data_root), args.capture_data_dir)
    total = 0
    for ep in episodes:
        if args.autoregressive:
            n = infer_episode_autoregressive(
                model=model,
                ep=ep,
                action_norm=action_norm,
                height=args.height,
                width=args.width,
                k=args.k,
                hist_stride=args.hist_stride,
                pred_horizon=args.pred_horizon,
                out_dir=out_dir,
                copy_gt=bool(args.copy_gt),
                start_t=args.start_t,
                max_steps_per_episode=args.max_steps_per_episode,
                camera_stride=args.camera_stride,
                device=device,
            )
        else:
            n = infer_episode(
                model=model,
                ep=ep,
                action_norm=action_norm,
                height=args.height,
                width=args.width,
                k=args.k,
                hist_stride=args.hist_stride,
                pred_horizon=args.pred_horizon,
                out_dir=out_dir,
                copy_gt=bool(args.copy_gt),
                max_frames_per_episode=args.max_frames_per_episode,
                device=device,
            )
        total += n

    print(f"Wrote {total} predictions to: {out_dir}")


if __name__ == "__main__":
    main()
