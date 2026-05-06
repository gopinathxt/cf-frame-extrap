from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def pose_from_pos_body_axes(
    pos_xyz: np.ndarray,
    body_x_axis_world: np.ndarray,
    body_y_axis_world: np.ndarray,
    body_z_axis_world: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build T_world_body (4x4) from position and body axes expressed in world frame.

    The capture CSVs store the drone/body axes in world coordinates (orn1/orn2/orn3).
    We re-orthonormalize to get a proper rotation matrix even if the logged axes drift.
    """
    x = _normalize(body_x_axis_world.astype(np.float32))

    # Gram–Schmidt y vs x
    y = body_y_axis_world.astype(np.float32)
    y = y - x * float(np.dot(x, y))
    y = _normalize(y)

    # Prefer cross(x,y) to enforce right-handedness
    z = np.cross(x, y)
    z = _normalize(z)
    y = np.cross(z, x)
    y = _normalize(y)

    R = np.stack([x, y, z], axis=1)  # columns are body axes in world frame
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = pos_xyz.astype(np.float32)
    return T


def extract_capture_zip(capture_zip: str | Path, out_dir: str | Path) -> List[Path]:
    """
    Extracts capture_*.csv files from a zip into out_dir (if needed) and returns sorted paths.
    """
    capture_zip = Path(capture_zip)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(out_dir.glob("capture_*.csv"))
    if csvs:
        return csvs

    with zipfile.ZipFile(capture_zip, "r") as zf:
        zf.extractall(out_dir)

    csvs = sorted(out_dir.glob("capture_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No capture_*.csv found after extracting {capture_zip} to {out_dir}")
    return csvs


def load_capture_csv_poses_actions(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      poses: [T,4,4] float32 (T_world_body)
      actions: [T,4] float32 from action1..action4
    """
    csv_path = Path(csv_path)
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float32)

    required = [
        "pos_x",
        "pos_y",
        "pos_z",
        "orn1_x",
        "orn1_y",
        "orn1_z",
        "orn2_x",
        "orn2_y",
        "orn2_z",
        "action1",
        "action2",
        "action3",
        "action4",
    ]
    missing = [c for c in required if c not in data.dtype.names]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}. Available: {list(data.dtype.names)}")

    pos = np.stack([data["pos_x"], data["pos_y"], data["pos_z"]], axis=1)
    xax = np.stack([data["orn1_x"], data["orn1_y"], data["orn1_z"]], axis=1)
    yax = np.stack([data["orn2_x"], data["orn2_y"], data["orn2_z"]], axis=1)

    # orn3 is optional; we can reconstruct it, but if present it helps sanity-checking upstream logs
    zax = None
    if all(c in data.dtype.names for c in ["orn3_x", "orn3_y", "orn3_z"]):
        zax = np.stack([data["orn3_x"], data["orn3_y"], data["orn3_z"]], axis=1)

    poses = np.stack(
        [pose_from_pos_body_axes(pos[i], xax[i], yax[i], None if zax is None else zax[i]) for i in range(pos.shape[0])],
        axis=0,
    )
    actions = np.stack([data["action1"], data["action2"], data["action3"], data["action4"]], axis=1).astype(np.float32)
    return poses.astype(np.float32), actions


def _sorted_frame_paths(frames_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [p for p in frames_dir.iterdir() if p.suffix.lower() in exts]
    paths.sort(key=lambda p: p.name)
    return paths


def _read_gray(path: Path) -> np.ndarray:
    # Be robust to RGB inputs: read as-is, then convert to grayscale if needed.
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 3:
        # OpenCV loads as BGR(A). Ignore alpha if present.
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img  # uint8 [H,W]


@dataclass(frozen=True)
class ActionNorm:
    mean: np.ndarray  # [4]
    std: np.ndarray  # [4]

    def normalize(self, a: np.ndarray) -> np.ndarray:
        return (a - self.mean) / (self.std + 1e-8)


def compute_action_norm(action_arrays: Sequence[np.ndarray]) -> ActionNorm:
    # action_arrays: list of [T,4]
    a = np.concatenate(action_arrays, axis=0).astype(np.float32)
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    std = np.maximum(std, 1e-3)
    return ActionNorm(mean=mean, std=std)


def build_augment(
    train: bool,
    height: int,
    width: int,
) -> A.ReplayCompose:
    # ReplayCompose lets us apply identical geometric transforms to a sequence.
    if not train:
        return A.ReplayCompose(
            [
                A.Resize(height=height, width=width, interpolation=cv2.INTER_AREA),
            ],
            additional_targets={},  # we’ll pass a list via explicit loop + replay
        )

    return A.ReplayCompose(
        [
            A.Resize(height=height, width=width, interpolation=cv2.INTER_AREA),
            # ShiftScaleRotate's params changed across Albumentations versions; Affine is the modern replacement.
            A.Affine(
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                scale=(0.95, 1.05),
                rotate=(-5, 5),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.8,
            ),
            A.Perspective(scale=(0.02, 0.06), keep_size=True, p=0.3),
            # Keep noise on, but avoid version-specific parameter names.
            A.GaussNoise(p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.15),
        ]
    )


class EpisodeIndex:
    def __init__(self, episode_dir: Path):
        self.episode_dir = episode_dir
        self.frames_dir = episode_dir / "frames"
        self.actions_path = episode_dir / "actions.npy"
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Missing frames dir: {self.frames_dir}")
        if not self.actions_path.exists():
            raise FileNotFoundError(f"Missing actions.npy: {self.actions_path}")

        self.frame_paths = _sorted_frame_paths(self.frames_dir)
        if len(self.frame_paths) == 0:
            raise FileNotFoundError(f"No frames found in: {self.frames_dir}")

        self.actions = np.load(self.actions_path).astype(np.float32)
        if self.actions.ndim != 2 or self.actions.shape[1] != 4:
            raise ValueError(f"actions.npy must have shape (T,4), got {self.actions.shape} in {self.actions_path}")

        t_img = len(self.frame_paths)
        t_act = self.actions.shape[0]
        if t_img != t_act:
            raise ValueError(
                f"Frame/action length mismatch in {episode_dir}: frames={t_img}, actions={t_act}. "
                "They must align by index."
            )

    @property
    def T(self) -> int:
        return len(self.frame_paths)


class RenderCaptureEpisode:
    """
    Episode backed by:
      - frames: gsplat_renders_all/gsplat_renders_capture_XX/*.png
      - actions: capture_data/capture_XX.csv (action1..action4)

    Exposes the same interface as EpisodeIndex (frame_paths, actions, T).
    """

    def __init__(self, frames_dir: Path, csv_path: Path):
        self.episode_dir = frames_dir
        self.frames_dir = frames_dir
        self.csv_path = csv_path

        self.frame_paths = _sorted_frame_paths(self.frames_dir)
        if len(self.frame_paths) == 0:
            raise FileNotFoundError(f"No frames found in: {self.frames_dir}")

        _poses, actions = load_capture_csv_poses_actions(self.csv_path)
        self.actions = actions.astype(np.float32)
        if self.actions.ndim != 2 or self.actions.shape[1] != 4:
            raise ValueError(f"Expected actions shape (T,4) from {self.csv_path}, got {self.actions.shape}")

        t_img = len(self.frame_paths)
        t_act = self.actions.shape[0]
        if t_img != t_act:
            raise ValueError(
                f"Frame/action length mismatch for {frames_dir.name}: frames={t_img}, actions={t_act}. "
                f"frames_dir={self.frames_dir} csv={self.csv_path}"
            )

    @property
    def T(self) -> int:
        return len(self.frame_paths)


def discover_episodes(data_root: str | Path) -> List[EpisodeIndex]:
    """
    Discovers episodes from a root directory, supporting multiple layouts.
    Layout 1 (preferred): data_root/episode_xxx/{frames/*.png, actions.npy}
    Layout 2 (fallback): data_root/{csv_dir}/{name}.csv, data_root/{renders_dir}/{name}/*.png
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")

    # Layout 1: Pre-processed episodes with actions.npy
    episodes: List[EpisodeIndex] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if p.is_dir() and (p / "frames").exists() and (p / "actions.npy").exists():
            try:
                episodes.append(EpisodeIndex(p))
            except (FileNotFoundError, ValueError):
                continue  # Skip invalid/empty episode dirs
    if episodes:
        return episodes

    # Layout 2: Renders and CSVs in separate directories
    # e.g., data_root/csv/*.csv and data_root/renders/name/*.png
    csv_dirs = [d for d in root.iterdir() if d.is_dir() and list(d.glob("*.csv"))]
    render_dirs = [d for d in root.iterdir() if d.is_dir() and list(d.glob("*/*.*"))]

    if not csv_dirs or not render_dirs:
        raise FileNotFoundError(
            f"No episodes found under {root}. Searched for pre-processed episodes "
            "and for separate csv/render directories, but found none."
        )

    # For simplicity, assume the first found directories are the correct ones.
    csv_dir = csv_dirs[0]
    renders_root = render_dirs[0]

    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        # Fallback for zipped CSVs, similar to original logic
        capture_zip = root / f"{csv_dir.name}.zip"
        if not capture_zip.exists():
            raise FileNotFoundError(f"No CSVs in {csv_dir} and no zip at {capture_zip}")
        csv_paths = extract_capture_zip(capture_zip, csv_dir)

    csv_by_key = {p.stem: p for p in csv_paths}

    render_eps: List[RenderCaptureEpisode] = []
    for frames_dir in sorted(renders_root.iterdir()):
        if not frames_dir.is_dir():
            continue
        key = frames_dir.name
        csv_path = csv_by_key.get(key)
        if csv_path is None:
            # Try to find a matching zip if CSVs were not found loose
            if not csv_paths and (root / f"{csv_dir.name}.zip").exists():
                csv_paths = extract_capture_zip(root / f"{csv_dir.name}.zip", csv_dir)
                csv_by_key = {p.stem: p for p in csv_paths}
                csv_path = csv_by_key.get(key)

        if csv_path:
            try:
                render_eps.append(RenderCaptureEpisode(frames_dir=frames_dir, csv_path=csv_path))
            except (FileNotFoundError, ValueError):
                continue  # Skip invalid/empty render dirs
        else:
            print(f"Warning: Missing CSV for render directory '{key}'. Skipping.")

    if not render_eps:
        raise FileNotFoundError(f"No valid render episodes found matching CSVs under {root}")

    return render_eps  # type: ignore[return-value]


def split_episodes(
    episodes: List[EpisodeIndex],
    val_ratio: float = 0.1,
    seed: int = 0,
) -> Tuple[List[EpisodeIndex], List[EpisodeIndex]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(episodes) * val_ratio)))
    val_idx = set(idx[:n_val].tolist())
    train, val = [], []
    for i, ep in enumerate(episodes):
        (val if i in val_idx else train).append(ep)
    return train, val


class FrameSequenceDataset(Dataset):
    """
    Samples tuples:
      frames_hist: [K,H,W] float32 in [0,1]
      actions_hist: [K,4] float32 normalized
      target: [1,H,W] float32 in [0,1]
    """

    def __init__(
        self,
        episodes: Sequence[EpisodeIndex],
        k: int = 3,
        height: int = 244,
        width: int = 324,
        train: bool = True,
        action_norm: Optional[ActionNorm] = None,
        hist_stride: int = 1,
        pred_horizon: int = 1,
    ):
        super().__init__()
        self.episodes = list(episodes)
        self.k = int(k)
        self.height = int(height)
        self.width = int(width)
        self.train = bool(train)
        self.aug = build_augment(train=train, height=height, width=width)
        self.hist_stride = int(hist_stride)
        self.pred_horizon = int(pred_horizon)
        if self.k != 3:
            raise ValueError("This dataset currently expects k=3 for spaced history [t-2s, t-s, t].")
        if self.hist_stride < 1:
            raise ValueError("hist_stride must be >= 1")
        if self.pred_horizon < 1:
            raise ValueError("pred_horizon must be >= 1")

        if action_norm is None:
            action_norm = compute_action_norm([ep.actions for ep in self.episodes])
        self.action_norm = action_norm

        # Build a flat index of (episode_id, t) where:
        #   history frames/actions at [t-2S, t-S, t]
        #   target frame at [t+H]
        # This matches your low-FPS camera use-case (sample sparse history, predict slightly ahead).
        self.samples: List[Tuple[int, int]] = []
        for ei, ep in enumerate(self.episodes):
            if ep.T <= (2 * self.hist_stride + self.pred_horizon):
                continue
            t_min = 2 * self.hist_stride
            t_max = ep.T - self.pred_horizon - 1
            for t in range(t_min, t_max + 1):
                self.samples.append((ei, t))
        if len(self.samples) == 0:
            raise ValueError("No valid samples found (need T > K in at least one episode).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ei, t = self.samples[idx]
        ep = self.episodes[ei]

        # load images
        s = self.hist_stride
        frame_paths = [ep.frame_paths[t - 2 * s], ep.frame_paths[t - s], ep.frame_paths[t]]
        target_path = ep.frame_paths[t + self.pred_horizon]

        frames_u8 = [_read_gray(p) for p in frame_paths]  # each [H,W] u8
        target_u8 = _read_gray(target_path)

        # apply identical augmentation to all frames + target via ReplayCompose
        replay = None
        frames_aug: List[np.ndarray] = []
        for i, fr in enumerate(frames_u8):
            if i == 0:
                out = self.aug(image=fr)
                replay = out["replay"]
                frames_aug.append(out["image"])
            else:
                out = A.ReplayCompose.replay(replay, image=fr)
                frames_aug.append(out["image"])
        target_aug = A.ReplayCompose.replay(replay, image=target_u8)["image"] if replay is not None else target_u8

        # normalize to [0,1]
        frames = np.stack(frames_aug, axis=0).astype(np.float32) / 255.0  # [K,H,W]
        target = target_aug.astype(np.float32)[None, ...] / 255.0  # [1,H,W]

        # actions history
        a_hist = np.stack([ep.actions[t - 2 * s], ep.actions[t - s], ep.actions[t]], axis=0)  # [K,4]
        a_hist = self.action_norm.normalize(a_hist).astype(np.float32)

        return {
            "frames": torch.from_numpy(frames),
            "actions": torch.from_numpy(a_hist),
            "target": torch.from_numpy(target),
        }

