from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _xavier_uniform_(t: Tensor) -> Tensor:
    nn.init.xavier_uniform_(t)
    return t


def _zeros_(t: Tensor) -> Tensor:
    nn.init.zeros_(t)
    return t


def _normal_002_(t: Tensor) -> Tensor:
    nn.init.normal_(t, mean=0.0, std=0.02)
    return t


def _linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    kernel_init=_xavier_uniform_,
    bias_init=_zeros_,
) -> nn.Linear:
    lin = nn.Linear(in_features, out_features, bias=bias)
    kernel_init(lin.weight)
    if bias:
        bias_init(lin.bias)
    return lin


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x32 = x.to(torch.float32)
        variance = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.eps)
        return (self.weight * x32).to(input_dtype)


class BottleneckTextProj(nn.Module):
    def __init__(self, text_encoder_dim: int, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.proj1 = _linear(text_encoder_dim, bottleneck_dim, bias=False)
        self.proj2 = _linear(bottleneck_dim, hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj2(self.proj1(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = _linear(
            frequency_embedding_size, hidden_size, bias=True,
            kernel_init=_normal_002_,
        )
        self.mlp_2 = _linear(
            hidden_size, hidden_size, bias=True,
            kernel_init=_normal_002_,
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.to(torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: Tensor) -> Tensor:
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        emb = self.mlp_0(emb)
        emb = F.silu(emb)
        emb = self.mlp_2(emb)
        return emb


def _rotate_half_interleaved(x: Tensor) -> Tensor:
    *prefix, d = x.shape
    x = x.reshape(*prefix, d // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    out = torch.stack((-x2, x1), dim=-1)
    return out.reshape(*prefix, d)


class TextRotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        pt_seq_len: int = 512,
        ft_seq_len: Optional[int] = None,
        theta: float = 10000.0,
        num_empty_token: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len if ft_seq_len is not None else pt_seq_len
        self.theta = theta
        self.num_empty_token = num_empty_token

        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim)
        )
        pos = torch.arange(self.ft_seq_len, dtype=torch.float32) / self.ft_seq_len * pt_seq_len
        freqs_main = torch.einsum("n,f->nf", pos, freqs)
        freqs_main = freqs_main.repeat_interleave(2, dim=-1)

        cos_main = torch.cos(freqs_main)
        sin_main = torch.sin(freqs_main)

        if num_empty_token > 0:
            cos_empty = torch.ones((num_empty_token, dim), dtype=torch.float32)
            sin_empty = torch.zeros((num_empty_token, dim), dtype=torch.float32)
            cos = torch.cat([cos_empty, cos_main], dim=0)
            sin = torch.cat([sin_empty, sin_main], dim=0)
        else:
            cos, sin = cos_main, sin_main

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        n = x.shape[-2]
        if n > self.cos.shape[0]:
            raise ValueError(
                f"RoPE sequence length {n} exceeds precomputed "
                f"{self.cos.shape[0]} (pt_seq_len + num_empty_token)."
            )
        cos = self.cos[:n].to(x.dtype)
        sin = self.sin[:n].to(x.dtype)
        return x * cos + _rotate_half_interleaved(x) * sin


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm_on = qk_norm
        self.attn_drop_p = attn_drop

        self.qkv = _linear(dim, dim * 3, bias=qkv_bias)
        self.proj = _linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x: Tensor,
        rope_fn: Optional[nn.Module] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)

        attn_bias: Optional[Tensor] = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                mask = attention_mask[:, None, :, :]
            else:
                mask = attention_mask
            attn_bias = torch.zeros_like(mask, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(mask == 0, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.proj_drop(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        adj_hidden = int(hidden_dim * 2 / 3)
        self.w12 = _linear(dim, 2 * adj_hidden, bias=bias)
        self.w3 = _linear(adj_hidden, dim, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        h = F.silu(x1) * x2
        h = self.drop(h)
        return self.w3(h)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = _linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            kernel_init=_zeros_,
            bias_init=_zeros_,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.norm_final(x))
