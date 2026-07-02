from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from elf_pytorch.layers import (
    Attention,
    BottleneckTextProj,
    FinalLayer,
    RMSNorm,
    SwiGLUFFN,
    TextRotaryEmbedding,
    TimestepEmbedder,
    _linear,
    _normal_002_,
    _xavier_uniform_,
    _zeros_,
)


class ELFBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(
        self,
        x: Tensor,
        rope_fn: Optional[nn.Module] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), rope_fn=rope_fn, attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ELF(nn.Module):

    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        vocab_size: int,
        *,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.bottleneck_dim = bottleneck_dim
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens

        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive (prefix time conditioning).")

        head_dim = hidden_size // num_heads

        self.self_cond_proj = _linear(
            2 * text_encoder_dim, text_encoder_dim, bias=True,
        )

        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)

        self.t_emb_tokens = nn.Parameter(
            _normal_002_(torch.empty(1, num_time_tokens, hidden_size))
        )
        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_tokens = nn.Parameter(
                _normal_002_(torch.empty(1, num_self_cond_cfg_tokens, hidden_size))
            )
        else:
            self.register_parameter("self_cond_cfg_tokens", None)

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(
                _normal_002_(torch.empty(1, num_model_mode_tokens, hidden_size))
            )
        else:
            self.register_parameter("mode_tokens", None)

        self.t_embedder = TimestepEmbedder(hidden_size)
        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
        else:
            self.self_cond_cfg_embedder = None

        prefix_len = num_time_tokens + num_self_cond_cfg_tokens + num_model_mode_tokens
        self.feat_rope = TextRotaryEmbedding(
            dim=head_dim,
            pt_seq_len=max_length,
            num_empty_token=prefix_len,
        )

        q1, q3 = depth // 4, depth // 4 * 3
        self.blocks = nn.ModuleList()
        for i in range(depth):
            in_drop_range = q3 > i >= q1
            self.blocks.append(
                ELFBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if in_drop_range else 0.0,
                    proj_drop=proj_drop if in_drop_range else 0.0,
                )
            )

        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=text_encoder_dim)

        self.proj_kernel = nn.Parameter(_xavier_uniform_(torch.empty(hidden_size, text_encoder_dim)))
        self.proj_bias = nn.Parameter(_zeros_(torch.empty(text_encoder_dim)))
        self.unembed_kernel = nn.Parameter(_xavier_uniform_(torch.empty(text_encoder_dim, vocab_size)))
        self.unembed_bias = nn.Parameter(_zeros_(torch.empty(vocab_size)))


    def _build_context(
        self, t: Tensor, self_cond_cfg_scale: Optional[Tensor], B: int
    ) -> Tensor:
        prefixes = []

        time_emb = self.t_embedder(t)
        time_prefix = self.t_emb_tokens.expand(B, -1, -1) + time_emb.unsqueeze(1)
        prefixes.append(time_prefix)

        if self_cond_cfg_scale is not None and self.self_cond_cfg_tokens is not None:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            sc_prefix = self.self_cond_cfg_tokens.expand(B, -1, -1) + sc_emb.unsqueeze(1)
            prefixes.append(sc_prefix)

        return torch.cat(prefixes, dim=1)


    def forward(
        self,
        x: Tensor,
        t: Tensor,
        attention_mask: Optional[Tensor] = None,
        self_cond_cfg_scale: Optional[Tensor] = None,
        decoder_step_active: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B = x.shape[0]
        text_C = self.text_encoder_dim

        if x.shape[-1] == 2 * text_C:
            x = self.self_cond_proj(x)
        elif x.shape[-1] != text_C:
            raise ValueError(
                f"Expected last dim to be {text_C} or {2 * text_C}, got {x.shape[-1]}"
            )

        x = self.text_proj(x)

        model_mode_offset = 0
        if self.num_model_mode_tokens > 0 and self.mode_tokens is not None:
            mode = self.mode_tokens.expand(B, -1, -1)
            active = (
                torch.zeros((), dtype=mode.dtype, device=mode.device)
                if decoder_step_active is None
                else decoder_step_active.to(dtype=mode.dtype)
            )
            mode = mode * active
            x = torch.cat([mode, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                pad = torch.ones((B, model_mode_offset), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([pad, attention_mask], dim=1)

        prefix = self._build_context(t, self_cond_cfg_scale, B)
        prefix_len = prefix.shape[1]
        x = torch.cat([prefix, x], dim=1)
        if attention_mask is not None:
            pad = torch.ones((B, prefix_len), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([pad, attention_mask], dim=1)

        for block in self.blocks:
            x = block(x, rope_fn=self.feat_rope, attention_mask=attention_mask)

        x = x[:, prefix_len + model_mode_offset:, :]

        denoised_output = self.final_layer(x)

        decoder_logits: Optional[Tensor] = None
        if decoder_step_active is not None:
            active_bool = (
                bool(decoder_step_active.item())
                if isinstance(decoder_step_active, Tensor) and decoder_step_active.numel() == 1
                else bool(decoder_step_active)
            )
            if active_bool:
                h = x @ self.proj_kernel + self.proj_bias
                h = nn.functional.gelu(h)
                decoder_logits = h @ self.unembed_kernel + self.unembed_bias

        return denoised_output, decoder_logits


def ELF_B(**kwargs) -> ELF:
    return ELF(depth=12, hidden_size=768, num_heads=12, **kwargs)


def ELF_M(**kwargs) -> ELF:
    return ELF(depth=24, hidden_size=1056, num_heads=16, **kwargs)


def ELF_L(**kwargs) -> ELF:
    return ELF(depth=32, hidden_size=1280, num_heads=16, **kwargs)


ELF_MODELS = {"ELF-B": ELF_B, "ELF-M": ELF_M, "ELF-L": ELF_L}
