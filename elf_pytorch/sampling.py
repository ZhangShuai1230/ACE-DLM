from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


def sample_timesteps(
    batch_size: int,
    *,
    P_mean: float = -0.8,
    P_std: float = 0.8,
    time_schedule: str = "logit_normal",
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if time_schedule == "logit_normal":
        z = torch.randn(batch_size, device=device, generator=generator) * P_std + P_mean
        return torch.sigmoid(z)
    if time_schedule == "uniform":
        return torch.rand(batch_size, device=device, generator=generator)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(
    n_steps: int,
    *,
    time_schedule: str = "logit_normal",
    P_mean: float = -0.8,
    P_std: float = 0.8,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    if time_schedule == "logit_normal":
        steps = sample_timesteps(
            n_steps - 1, P_mean=P_mean, P_std=P_std,
            time_schedule=time_schedule, device=device, generator=generator,
        )
        steps, _ = torch.sort(steps)
        return torch.cat([torch.tensor([0.0], device=device),
                          steps,
                          torch.tensor([1.0], device=device)])
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def sample_cfg_scale(
    batch_size: int,
    *,
    cfg_min: float = 0.0,
    cfg_max: float = 3.0,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    u = torch.rand(batch_size, device=device, generator=generator)
    a = 1.0 + cfg_min
    b = 1.0 + cfg_max
    return a * torch.exp(u * math.log(b / a)) - 1.0


def restore_cond(z_updated: Tensor, cond_seq: Tensor, cond_seq_mask: Tensor) -> Tensor:
    mask = cond_seq_mask
    target_ndim = max(z_updated.ndim, cond_seq.ndim)
    while mask.ndim < target_ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(v: Tensor, x: Tensor, cond_seq: Optional[Tensor], cond_seq_mask: Optional[Tensor]):
    if cond_seq is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


def add_noise(
    x0: Tensor, noise: Tensor, t: Tensor, *, denoiser_noise_scale: float = 1.0,
    cond_seq_mask: Optional[Tensor] = None,
) -> Tensor:
    t_expanded = t.reshape(-1, 1, 1)
    z = t_expanded * x0 + (1.0 - t_expanded) * noise * denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1.0 - cond_seq_mask) * z
    return z


def net_out_to_v_x(
    net_out, z: Tensor, t: Tensor, t_eps: float = 5e-2
) -> Tuple[Tensor, Tensor]:
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = t.reshape(-1, 1, 1)
    x = net_out
    v = (x - z) / torch.clamp(1.0 - t_reshaped, min=t_eps)
    return v, x


@dataclass
class SamplingArgs:
    t_eps: float = 5e-2
    self_cond_prob: float = 0.5
    num_self_cond_cfg_tokens: int = 4
    denoiser_noise_scale: float = 1.0
    cfg_scale: float = 1.0
    self_cond_cfg_scale: float = 1.0


def _forward_self_cond(
    model,
    z: Tensor,
    t_batch: Tensor,
    x_pred_prev: Optional[Tensor],
    args: SamplingArgs,
    cond_seq: Optional[Tensor],
    cond_seq_mask: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    t_eps = args.t_eps
    w = args.self_cond_cfg_scale

    def restore(v, x):
        if cond_seq is not None:
            return restore_vx(v, x, cond_seq, cond_seq_mask)
        return v, x

    if args.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = torch.zeros_like(z)
            if cond_seq is not None:
                x_pred_prev = restore_cond(x_pred_prev, cond_seq, cond_seq_mask)
        z_input = torch.cat([z, x_pred_prev], dim=-1)
        w_batch = torch.full((z.shape[0],), w, device=z.device, dtype=z.dtype)
        net_out = model(z_input, t_batch, self_cond_cfg_scale=w_batch)
        v_cond, x_cond = net_out_to_v_x(net_out, z, t_batch, t_eps)
        return restore(v_cond, x_cond)

    if args.self_cond_prob == 0:
        net_out = model(z, t_batch)
        v, x = net_out_to_v_x(net_out, z, t_batch, t_eps)
        return restore(v, x)

    if x_pred_prev is None or w == 0.0:
        z_input_uncond = torch.cat([z, torch.zeros_like(z)], dim=-1)
        if cond_seq is not None:
            z_input_uncond = torch.cat(
                [z, restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)], dim=-1
            )
        net_out_uncond = model(z_input_uncond, t_batch)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, t_eps)
        return restore(v_uncond, x_uncond)
    z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
    net_out_cond = model(z_input_cond, t_batch)
    v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
    if w == 1.0:
        return restore(v_cond, x_cond)
    z_input_uncond = torch.cat([z, torch.zeros_like(z)], dim=-1)
    net_out_uncond = model(z_input_uncond, t_batch)
    v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, t_eps)
    v = v_uncond + w * (v_cond - v_uncond)
    x = x_uncond + w * (x_cond - x_uncond)
    return restore(v, x)


def _forward_sample(
    model,
    z: Tensor,
    t_batch: Tensor,
    x_pred_prev: Optional[Tensor],
    args: SamplingArgs,
    cond_seq: Optional[Tensor],
    cond_seq_mask: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    v_cond, x_cond = _forward_self_cond(
        model, z, t_batch, x_pred_prev, args, cond_seq, cond_seq_mask,
    )
    if args.cfg_scale == 1.0 or cond_seq is None:
        return v_cond, x_cond

    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
    if x_pred_prev is not None:
        x_pred_prev_uncond = restore_cond(
            x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask
        )
    else:
        x_pred_prev_uncond = None
    v_uncond, x_uncond = _forward_self_cond(
        model, z_uncond, t_batch, x_pred_prev_uncond, args,
        cond_seq=torch.zeros_like(cond_seq), cond_seq_mask=cond_seq_mask,
    )
    w = args.cfg_scale
    v = v_uncond + w * (v_cond - v_uncond)
    x = x_uncond + w * (x_cond - x_uncond)
    return restore_vx(v, x, cond_seq, cond_seq_mask)


@torch.no_grad()
def ode_step(
    model,
    z: Tensor, t: float, t_next: float,
    x_pred_prev: Optional[Tensor],
    args: SamplingArgs,
    cond_seq: Optional[Tensor] = None,
    cond_seq_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    t_batch = torch.full((z.shape[0],), float(t), device=z.device, dtype=z.dtype)
    v_pred, x_pred = _forward_sample(
        model, z, t_batch, x_pred_prev, args, cond_seq, cond_seq_mask,
    )
    z_new = z + (t_next - t) * v_pred
    return z_new, x_pred


@torch.no_grad()
def sde_step(
    model,
    z: Tensor, t: float, t_next: float,
    x_pred_prev: Optional[Tensor],
    args: SamplingArgs,
    gamma: float,
    generator: Optional[torch.Generator] = None,
    cond_seq: Optional[Tensor] = None,
    cond_seq_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    h = t_next - t
    alpha = max(0.0, min(1.0, 1.0 - gamma * h))
    t_back = alpha * t
    eps = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator) * args.denoiser_noise_scale
    z_back = alpha * z + (1.0 - alpha) * eps
    if cond_seq is not None:
        z_back = restore_cond(z_back, cond_seq, cond_seq_mask)
    t_batch = torch.full((z.shape[0],), float(t_back), device=z.device, dtype=z.dtype)
    v_pred, x_pred = _forward_sample(
        model, z_back, t_batch, x_pred_prev, args, cond_seq, cond_seq_mask,
    )
    z_new = z_back + (t_next - t_back) * v_pred
    return z_new, x_pred


@torch.no_grad()
def generate_samples(
    model,
    z: Tensor,
    t_steps: Tensor,
    args: SamplingArgs,
    method: str = "sde",
    sde_gamma: float = 1.5,
    generator: Optional[torch.Generator] = None,
    cond_seq: Optional[Tensor] = None,
    cond_seq_mask: Optional[Tensor] = None,
) -> Tensor:
    model.eval()
    if cond_seq is None:
        cond_seq_mask_3d = None
    else:
        if cond_seq_mask.ndim == 2:
            cond_seq_mask = cond_seq_mask.unsqueeze(-1)
        cond_seq_mask_3d = cond_seq_mask
        z = restore_cond(z, cond_seq, cond_seq_mask)

    x_pred: Optional[Tensor] = None
    if cond_seq is not None:
        x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    n_steps = t_steps.shape[0] - 1

    for i in range(n_steps - 1):
        t = float(t_steps[i].item())
        t_next = float(t_steps[i + 1].item())
        if method == "sde":
            z, x_pred = sde_step(
                model, z, t, t_next, x_pred, args,
                gamma=sde_gamma, generator=generator,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_3d,
            )
        elif method == "ode":
            z, x_pred = ode_step(
                model, z, t, t_next, x_pred, args,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_3d,
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    t = float(t_steps[-2].item())
    t_next = float(t_steps[-1].item())
    z, _ = ode_step(
        model, z, t, t_next, x_pred, args,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_3d,
    )
    return z


@torch.no_grad()
def decode_latents_to_tokens(
    model,
    z: Tensor,
    args: SamplingArgs,
    t_final: float = 1.0,
) -> Tensor:
    model.eval()
    t = torch.full((z.shape[0],), float(t_final), device=z.device, dtype=z.dtype)
    w = torch.full(
        (z.shape[0],), float(args.self_cond_cfg_scale), device=z.device, dtype=z.dtype,
    ) if args.num_self_cond_cfg_tokens > 0 else None

    if args.self_cond_prob > 0:
        z_input = torch.cat([z, torch.zeros_like(z)], dim=-1)
    else:
        z_input = z
    _, logits = model(
        z_input, t,
        self_cond_cfg_scale=w,
        decoder_step_active=torch.tensor(True),
    )
    return logits.argmax(dim=-1)


def mask_after_eos(predicted_ids: Tensor, eos_token_id: int, pad_token_id: int) -> Tensor:
    eos = (predicted_ids == eos_token_id)
    keep = (eos.cumsum(dim=1) == 0)
    return torch.where(keep, predicted_ids, torch.full_like(predicted_ids, pad_token_id))
