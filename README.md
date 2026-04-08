# cf-frame-extrap

Action-conditioned frame extrapolation for low-FPS onboard vision.

Goal: given a short history of grayscale frames and actions (**CTBR**, 4 floats), predict a future frame. This repo supports a “spaced history” mode that matches low-FPS cameras: use frames \([t-12,\ t-6,\ t]\) to predict \(t+2\).

## Dataset layout (recommended)

Create a dataset root with episode folders:

```
data_root/
  episode_000/
    frames/
      000000.png
      000001.png
      ...
    actions.npy          # shape: (T, 4) float32, columns: [C, T, B, R]
  episode_001/
    frames/...
    actions.npy
```

Notes:
- Frames must be grayscale (or will be converted to grayscale).
- Frames are expected to be in chronological order by filename.
- `actions.npy` must align with frames by index.

## Quickstart

Create environment (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `opencv-python` fails to install on your machine, install OS deps (e.g. Ubuntu: `sudo apt-get install -y libgl1`) or swap OpenCV for your preferred image IO backend.

Train (spaced history, recommended for low-FPS camera setup):

```bash
python -m cf_frame_extrap.train \
  --data_root /path/to/data_root \
  --k 3 \
  --hist_stride 6 \
  --pred_horizon 2 \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 4 \
  --out_dir runs/exp_s6_h2
```

Teacher-forced inference (uses your existing frames/actions as input history, predicts and saves targets):

```bash
python -m cf_frame_extrap.infer_dataset \
  --ckpt runs/exp_s6_h2/best.pt \
  --data_root /path/to/data_root \
  --out_dir runs/exp_s6_h2/teacher_forced \
  --k 3 \
  --hist_stride 6 \
  --pred_horizon 2 \
  --save_gt
```

Rollout (predict forward indefinitely from an initial buffer; useful for online frame synthesis):

```bash
python -m cf_frame_extrap.infer_rollout \
  --ckpt runs/exp_s6_h2/best.pt \
  --k 3 \
  --init_frames_dir /path/to/init_frames \
  --actions_npy /path/to/actions.npy \
  --out_dir runs/exp_s6_h2/rollout_preview
```

Single-step prediction from exactly 3 frames + 3 actions:

```bash
python -m cf_frame_extrap.predict_one \
  --ckpt runs/exp_s6_h2/best.pt \
  --img f0.png f1.png f2.png \
  --action  C0 T0 B0 R0   C1 T1 B1 R1   C2 T2 B2 R2 \
  --out pred.png
```

## Model

- **Backbone**: small UNet-like encoder/decoder operating on stacked frames (channels = K).
- **Action conditioning**: MLP embedding of action history (K×4) injected as FiLM (scale/shift) in multiple blocks.
- **Output**: next grayscale frame in \([0,1]\).

## Loss (stable default)

Reconstruction-only (recommended first):
- L1 loss
- SSIM loss (1 - SSIM)
- Gradient difference loss (keeps edges/gates crisp)

Once this is working, you can add a GAN head later if you want extra sharpness.

