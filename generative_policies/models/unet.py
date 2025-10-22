from typing import Union

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from abc import ABC, abstractmethod
from generative_policies.utils import SinusoidalPosEmb, GaussianFourierEmb

class ConditionalUNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, diffusion_step_embed_dim, down_dims, cond_dim):
        super(ConditionalUNet1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.down_dims = down_dims
        self.cond_dim = cond_dim
        
        # Time embedding (Gaussian Fourier + MLP)
        self.time_embed = nn.Sequential(
            GaussianFourierEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.LayerNorm(diffusion_step_embed_dim * 4),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
            nn.LayerNorm(diffusion_step_embed_dim)
        )
        
        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, diffusion_step_embed_dim),
            nn.SiLU(),
            nn.LayerNorm(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
            nn.LayerNorm(diffusion_step_embed_dim)
        )
        
        # Condition projection layers for each encoder/decoder layer
        self.cond_projections = nn.ModuleList()
        for dim in down_dims:
            self.cond_projections.append(
                nn.Linear(diffusion_step_embed_dim, dim)
            )
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, down_dims[0]),
            nn.LayerNorm(down_dims[0])
        )
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(len(down_dims)):
            if i == 0:
                # First layer: input_proj output + condition embedding
                in_ch = down_dims[i] + down_dims[i]  # h + cond_emb_proj
            else:
                # Subsequent layers: previous layer output + condition embedding
                in_ch = down_dims[i-1] + down_dims[i]  # h + cond_emb_proj
            out_ch = down_dims[i]
            self.encoder_layers.append(
                nn.Sequential(
                    nn.LayerNorm(in_ch),
                    nn.Linear(in_ch, out_ch),
                    nn.SiLU(),
                    nn.LayerNorm(out_ch),
                    nn.Linear(out_ch, out_ch),
                    nn.LayerNorm(out_ch)
                )
            )
        
        # Decoder layers (reverse order of encoder)
        self.decoder_layers = nn.ModuleList()
        for i in range(len(down_dims)):
            # Decoder works in reverse order: [512, 256, 128] for down_dims=[128, 256, 512]
            decoder_idx = len(down_dims) - 1 - i  # Reverse index
            
            if i == 0:
                # First decoder layer: last encoder output + skip connection + condition embedding
                in_ch = down_dims[-1] + down_dims[decoder_idx] + down_dims[decoder_idx]  # h + skip + cond_emb_proj
            else:
                # Subsequent decoder layers: previous decoder output + skip connection + condition embedding
                prev_decoder_idx = len(down_dims) - i  # Previous decoder output dimension
                in_ch = down_dims[prev_decoder_idx] + down_dims[decoder_idx] + down_dims[decoder_idx]  # h + skip + cond_emb_proj
            out_ch = down_dims[decoder_idx]
            self.decoder_layers.append(
                nn.Sequential(
                    nn.LayerNorm(in_ch),
                    nn.Linear(in_ch, out_ch),
                    nn.SiLU(),
                    nn.LayerNorm(out_ch),
                    nn.Linear(out_ch, out_ch),
                    nn.LayerNorm(out_ch)
                )
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(down_dims[0]),
            nn.Linear(down_dims[0], out_channels)
        )

    def forward(self, x, c, t):
        """
        x: (batch_size, flow_dim)
        c: (batch_size, cond_dim)
        t: (batch_size, 1) - time steps
        return: (batch_size, flow_dim)
        """

        batch_size = x.shape[0]
        
        # Embed time and condition
        # Accept t as (B,1) or (B,) floats
        t_inp = t.squeeze(-1) if t.dim() == 2 and t.size(-1) == 1 else t
        t_emb = self.time_embed(t_inp)  # (batch_size, diffusion_step_embed_dim)
        c_emb = self.cond_embed(c)  # (batch_size, diffusion_step_embed_dim)
        
        # Combine time and condition embeddings
        cond_emb = t_emb + c_emb  # (batch_size, diffusion_step_embed_dim)
        
        # Input projection
        h = self.input_proj(x)  # (batch_size, down_dims[0])
        
        # Encoder
        encoder_outputs = []
        for i, layer in enumerate(self.encoder_layers):
            # Project condition embedding to match current layer dimension
            cond_emb_proj = self.cond_projections[i](cond_emb)
            h = layer(torch.cat([h, cond_emb_proj], dim=-1))
            encoder_outputs.append(h)
        
        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            # Decoder works in reverse order, so we need to use reverse index for condition projection
            decoder_idx = len(self.down_dims) - 1 - i
            
            # Get skip connection from encoder
            skip = encoder_outputs[decoder_idx]
            
            # Project condition embedding to match current layer dimension
            cond_emb_proj = self.cond_projections[decoder_idx](cond_emb)
            
            # Concatenate: current hidden state + skip connection + condition
            h = layer(torch.cat([h, skip, cond_emb_proj], dim=-1))
        
        # Output projection
        output = self.output_proj(h)
        
        return output
    
    def to(self, device):
        super().to(device)
        return self