# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

#########################################################
#                   Random stuff from timm              #
#########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
import collections
from itertools import repeat
from typing import Final, Optional, Type, Union, Tuple
from functools import partial

_autowrap_functions = set()

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def register_notrace_function(name_or_fn):
    _autowrap_functions.add(name_or_fn)
    return name_or_fn

@torch.fx.wrap
@register_notrace_function
def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask
# Set to True if exporting a model with Same padding via ONNX
_EXPORTABLE = False

_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

if 'TIMM_FUSED_ATTN' in os.environ:
    _USE_FUSED_ATTN = int(os.environ['TIMM_FUSED_ATTN'])
else:
    _USE_FUSED_ATTN = 1  # 0 == off, 1 == on (for tested use), 2 == on (for experimental use)


def use_fused_attn(experimental: bool = False) -> bool:
    # NOTE: ONNX export cannot handle F.scaled_dot_product_attention as of pytorch 2.0
    if not _HAS_FUSED_ATTN or _EXPORTABLE:
        return False
    if experimental:
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
            device=None,
            dtype=None
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, **dd)
        self.q_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Optional[Type[nn.Module]] = None,
            bias: Union[bool, Tuple[bool, bool]] = True,
            drop: Union[float, Tuple[float, float]] = 0.,
            use_conv: bool = False,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0], **dd)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features, **dd) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1], **dd)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        cond_dim,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads

        self.time_embed = TimestepEmbedder(hidden_size)
        
        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Input projection (replaces patch embedding)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, out_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize input projection:
        nn.init.xavier_uniform_(self.input_proj[0].weight)
        nn.init.constant_(self.input_proj[0].bias, 0)

        # Initialize condition embedding:
        nn.init.normal_(self.cond_embed[0].weight, std=0.02)
        nn.init.constant_(self.cond_embed[0].bias, 0)
        nn.init.normal_(self.cond_embed[3].weight, std=0.02)
        nn.init.constant_(self.cond_embed[3].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, c, t):
        """
        Forward pass of DiT adapted for states.
        x: (batch_size, in_dim) tensor of state inputs
        c: (batch_size, cond_dim) tensor of conditioning inputs
        t: (batch_size, 1) - time steps
        return: (batch_size, out_dim)
        """
        # Input projection (replaces patch embedding)
        x = self.input_proj(x)  # (batch_size, hidden_size)
        
        # Time embedding
        # Accept t as (B,1) or (B,) floats
        t_inp = t.squeeze(-1) if t.dim() == 2 and t.size(-1) == 1 else t
        t_emb = self.time_embed(t_inp)  # (batch_size, hidden_size)
        
        # Condition embedding
        c_emb = self.cond_embed(c)  # (batch_size, hidden_size)
        
        # Combine time and condition embeddings
        cond_emb = t_emb + c_emb  # (batch_size, hidden_size)
        
        # Add sequence dimension for transformer blocks
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Apply transformer blocks (cond_emb stays 2D for adaLN modulation)
        for block in self.blocks:
            x = block(x, cond_emb)  # (batch_size, 1, hidden_size)
        
        # Final layer (cond_emb stays 2D for adaLN modulation)
        x = self.final_layer(x, cond_emb)  # (batch_size, 1, out_dim)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (batch_size, out_dim)
        
        return x

    def forward_with_cfg(self, x, c, t, cfg_scale=1.0):
        """
        Forward pass of DiT with classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, c, t)
        
        # Apply classifier-free guidance
        if self.learn_sigma:
            eps, rest = model_out[:, :self.in_dim], model_out[:, self.in_dim:]
        else:
            eps, rest = model_out, None
            
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        
        if rest is not None:
            return torch.cat([eps, rest], dim=1)
        else:
            return eps


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiT_S(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL': DiT_XL,
    'DiT-L':  DiT_L,
    'DiT-B':  DiT_B,
    'DiT-S':  DiT_S,
}