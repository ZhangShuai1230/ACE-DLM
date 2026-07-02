"""Self-conditioning steering: estimate the difference-of-means direction d and subtract
lambda*d from the self-conditioning feedback at every sampling step.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from elf_pytorch.model import ELF_MODELS
from elf_pytorch.sampling import (SamplingArgs, get_sampling_steps,
                                  sde_step, ode_step, mask_after_eos)
from transformers import AutoTokenizer
from tqdm.auto import tqdm

STEPS, GAMMA, SC_CFG, NOISE, P_MEAN, P_STD = 64, 1.0, 3.0, 2.0, -1.5, 0.8


def build_elf(size="B", ckpt=None, device="cuda", dtype=torch.float32):
    "Build ELF and load weights"
    m = ELF_MODELS[f"ELF-{size}"](
        text_encoder_dim=512, max_length=1024, vocab_size=32100,
        num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
        bottleneck_dim=128,
    ).eval()
    if ckpt:
        ckpt = _resolve_ckpt(ckpt)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "ema_params1" in sd:
            sd = sd["ema_params1"]   # EMA weights
        m.load_state_dict(sd, strict=True)
    return m.to(device=device, dtype=dtype)


def _resolve_ckpt(ckpt):
    p = Path(ckpt)
    if not p.is_dir():
        return ckpt 
    hits = sorted(c for c in p.glob("checkpoint_*") if c.is_file())
    if not hits:
        hits = sorted((f for f in p.iterdir()
                       if f.is_file() and not f.name.startswith(".")
                       and f.suffix not in (".metadata", ".lock")),
                      key=lambda f: f.stat().st_size)
    if not hits:
        raise FileNotFoundError(f"no weight file found in checkpoint dir {p}")
    return str(hits[-1])   # the unique / largest weight file


def load_direction(path, device="cuda"):
    """Load a steering direction (dict with key 'd', or a bare tensor) onto `device`."""
    blob = torch.load(path, map_location=device)
    d = blob["d"] if isinstance(blob, dict) else blob
    return d.to(device)


def sampling_args(sc_cfg=SC_CFG, noise=NOISE):
    return SamplingArgs(t_eps=0.05, self_cond_prob=0.5, num_self_cond_cfg_tokens=4,
                        denoiser_noise_scale=noise, cfg_scale=1.0, self_cond_cfg_scale=sc_cfg)


@torch.no_grad()
def _decode_ids(model, z, args, t_final, eos, pad):
    t = torch.full((z.shape[0],), float(t_final), device=z.device, dtype=z.dtype)
    w = torch.full((z.shape[0],), float(args.self_cond_cfg_scale), device=z.device,
                   dtype=z.dtype) if args.num_self_cond_cfg_tokens > 0 else None
    z_in = torch.cat([z, torch.zeros_like(z)], dim=-1) if args.self_cond_prob > 0 else z
    _, logits = model(z_in, t, self_cond_cfg_scale=w, decoder_step_active=torch.tensor(True))
    return mask_after_eos(logits.argmax(dim=-1), eos_token_id=eos, pad_token_id=pad)


@torch.no_grad()
def _run(model, z, t_steps, args, *, gamma=GAMMA, generator=None, method="sde",
         steer_d=None, lam=0.0, collect=None):
    """SDE/ODE trajectory with optional steered self-conditioning feedback."""
    model.eval()
    x_pred: Optional[torch.Tensor] = None
    n = t_steps.shape[0] - 1
    for i in range(n - 1):
        t, t_next = float(t_steps[i]), float(t_steps[i + 1])
        if method == "ode":
            z, x_pred = ode_step(model, z, t, t_next, x_pred, args)
        else:
            z, x_pred = sde_step(model, z, t, t_next, x_pred, args, gamma=gamma, generator=generator)
        if steer_d is not None and lam != 0.0:
            x_pred = x_pred - lam * steer_d.view(1, 1, -1)
        if collect is not None:
            collect["sum"] = collect.get("sum", 0) + x_pred.mean(dim=1)
            collect["count"] = collect.get("count", 0) + 1
    z, _ = ode_step(model, z, float(t_steps[-2]), float(t_steps[-1]), x_pred, args)
    return z


@torch.no_grad()
def estimate_direction(model, n=400, *, tokenizer=None, args=None, device="cuda", bs=30, seed=42):
    """Difference-of-means basin direction d (unit vector). Collects the mean SC feedback
    per trajectory, sorts by seq-rep-4, and returns mean(top tertile) - mean(bottom tertile)."""
    from metrics import seq_rep_4
    tok = tokenizer or AutoTokenizer.from_pretrained("t5-small")
    pad, eos = tok.pad_token_id or 0, tok.eos_token_id or 1
    args = args or sampling_args()
    gen = torch.Generator(device=device).manual_seed(seed)
    feats, reps = [], []
    pbar = tqdm(total=n, desc="estimate d")
    for bi in range((n + bs - 1) // bs):
        cur = min(bs, n - bi * bs); acc = {}
        ts = get_sampling_steps(STEPS, time_schedule="logit_normal", P_mean=P_MEAN, P_std=P_STD,
                                device=device, generator=gen).to(torch.float32)
        z = torch.randn(cur, 1024, 512, device=device, generator=gen) * NOISE
        zf = _run(model, z, ts, args, generator=gen, collect=acc)
        avg = (acc["sum"] / acc["count"]).cpu()
        ids = _decode_ids(model, zf, args, float(ts[-1]), eos, pad)
        for i in range(cur):
            feats.append(avg[i]); reps.append(seq_rep_4(tok.decode(ids[i].cpu().tolist(), skip_special_tokens=True)))
        pbar.update(cur)
    pbar.close()
    S = torch.stack(feats); r = torch.tensor(reps)
    o = torch.argsort(r); t = len(o) // 3
    d = S[o[-t:]].mean(0) - S[o[:t]].mean(0)
    return (d / (d.norm() + 1e-8)).to(device)


@torch.no_grad()
def generate(model, n=500, *, steer_d=None, lam=0.0, tokenizer=None, args=None,
             device="cuda", bs=30, steps=STEPS, method="sde", gamma=GAMMA, seed=42,
             generator=None):
    "Generate n unconditional samples; if steer_d/lam given, apply steered SC."
    tok = tokenizer or AutoTokenizer.from_pretrained("t5-small")
    pad, eos = tok.pad_token_id or 0, tok.eos_token_id or 1
    args = args or sampling_args()
    gen = generator if generator is not None else torch.Generator(device=device).manual_seed(seed)
    sd = steer_d.to(device) if steer_d is not None else None
    texts = []
    pbar = tqdm(total=n, desc="sample")
    for bi in range((n + bs - 1) // bs):
        cur = min(bs, n - bi * bs)
        ts = get_sampling_steps(steps, time_schedule="logit_normal", P_mean=P_MEAN, P_std=P_STD,
                                device=device, generator=gen).to(torch.float32)
        z = torch.randn(cur, 1024, 512, device=device, generator=gen) * NOISE
        zf = _run(model, z, ts, args, gamma=gamma, method=method, steer_d=sd, lam=lam, generator=gen)
        ids = _decode_ids(model, zf, args, float(ts[-1]), eos, pad)
        for i in range(cur):
            texts.append(tok.decode(ids[i].cpu().tolist(), skip_special_tokens=True))
        pbar.update(cur)
    pbar.close()
    return texts
