from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g


def _create_ssim_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    g = _gaussian_kernel(window_size, sigma, device, dtype)
    window_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]
    window = window_2d.repeat(channels, 1, 1, 1)  # [C,1,ws,ws] for groups=C conv
    return window


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    x,y: [B,C,H,W] in [0,1]
    returns: [B] SSIM
    """
    assert x.shape == y.shape
    b, c, _, _ = x.shape
    device, dtype = x.device, x.dtype
    window = _create_ssim_window(window_size, sigma, c, device, dtype)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=c) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + eps)
    return ssim_map.view(b, c, -1).mean(dim=(1, 2))


def gradient_difference_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred/target: [B,1,H,W]
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()


def recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    w_l1: float = 1.0,
    w_ssim: float = 0.5,
    w_gdl: float = 0.2,
) -> torch.Tensor:
    l1 = F.l1_loss(pred, target)
    s = ssim(pred, target).mean()
    gdl = gradient_difference_loss(pred, target)
    return w_l1 * l1 + w_ssim * (1.0 - s) + w_gdl * gdl

