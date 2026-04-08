# cf-frame-extrap

Action-conditioned frame extrapolation for low-FPS onboard vision.

Goal: given the last **K** grayscale frames (default **K=3**) and the last **K** actions (**CTBR**, 4 floats), predict the next frame. At runtime, you can roll the model forward to synthesize intermediate frames to feed a vision-based RL policy at a higher effective FPS.

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

Train:

```bash
python -m cf_frame_extrap.train \
  --data_root /path/to/data_root \
  --k 3 \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 4 \
  --out_dir runs/exp1
```

Rollout (predict forward indefinitely from an initial buffer):

```bash
python -m cf_frame_extrap.infer_rollout \
  --ckpt runs/exp1/best.pt \
  --k 3 \
  --init_frames_dir /path/to/init_frames \
  --actions_npy /path/to/actions.npy \
  --out_dir runs/exp1/rollout_preview
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

