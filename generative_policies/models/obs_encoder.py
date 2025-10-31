from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from generative_policies.utils import inputs_to_torch

class ImageEncoder(nn.Module):
    def __init__(self, input_dim: int, device: str = 'cuda'):
        super().__init__()
        from torchvision.models import resnet50
        import torchvision.transforms as transforms
        self.input_dim = input_dim
        self.encoder = resnet50().eval()
        self.encoder.fc = nn.Identity()
        # ImageNet normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
        self.to(device)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224, device=device)
            self._output_dim = self.encoder(dummy_input).shape[-1]

    def forward(self, input_data):
        input_data = inputs_to_torch(input_data, self.device)
        
        if len(input_data.shape) == 3:
            input_data = input_data.unsqueeze(0)
        elif len(input_data.shape) == 5:
            input_data = input_data.squeeze(dim=1)
        
        # PIL to torch tensor [B, H, W, C] -> [B, C, H, W]
        input_data = input_data.permute(0, 3, 1, 2).float() / 255.0
        
        # Resize to 224x224 and normalize
        input_data = torch.nn.functional.interpolate(input_data, size=(224, 224), mode='bilinear', align_corners=False)
        input_data = self.normalize(input_data)
        
        with torch.no_grad():
            embedding = self.encoder(input_data)
        return embedding

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        return self

    @property
    def output_dim(self):
        return self._output_dim


class ImageStateEncoder(nn.Module):
    def __init__(self, image_dim: int, state_dim: int, device: str = 'cuda'):
        super().__init__()
        self.image_encoder = ImageEncoder(image_dim, device=device)
        self.state_encoder = IdentityEncoder(state_dim, device=device)

    def forward(self, input_data):
        image_data, state_data = input_data
        image_data = self.image_encoder(image_data)
        state_data = self.state_encoder(state_data)
        return torch.cat([image_data, state_data], dim=-1)
    
    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.state_encoder.to(device)
        return self

    @property
    def output_dim(self):
        return self.image_encoder.output_dim + self.state_encoder.output_dim

class IdentityEncoder(nn.Module):
    """
    Handles observations that are numpy arrays or torch tensors 
    of shapes (B, D) or (B, 1, D) or (D)
    puts them on the correct device
    """

    def __init__(self, input_dim: int, device: str = 'cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def forward(self, input_data):
        input_data = inputs_to_torch(input_data, self.device)

        if len(input_data.shape) == 1: # no batch provided
            input_data = input_data.unsqueeze(0)
        elif len(input_data.shape) == 3: # seq dim 1 provided
            assert input_data.shape[1] == 1, "No sequence length expected"
            input_data = input_data.squeeze(dim=1)

        input_data = input_data.to(self.device)
        
        return input_data

    @property
    def output_dim(self):
        return self.input_dim
    
class IdentityTwoInputEncoder(nn.Module):
    """
    Encodes two input tensors/arrays into a single tensor, handling batching and device
    Useful for multiple conditioning variables (e.g. obs and action) 
    """
    
    def __init__(self, input_1_dim, input_2_dim, device='cuda'):
        super(IdentityTwoInputEncoder, self).__init__()
        self.input_1_dim = input_1_dim
        self.input_2_dim = input_2_dim
        self.input_1_encoder = IdentityEncoder(input_1_dim, device=device)
        self.input_2_encoder = IdentityEncoder(input_2_dim, device=device)
        self.device = device

    def forward(self, input_1, input_2) -> torch.Tensor:
        input_1_cond = self.input_1_encoder(input_1)
        input_2_cond = self.input_2_encoder(input_2)
        cond = torch.cat([input_1_cond, input_2_cond], dim=-1)
        return cond

    @property
    def output_dim(self):
        return self.input_1_dim + self.input_2_dim

    def to(self, device):
        self.device = device
        self.input_1_encoder.to(device)
        self.input_2_encoder.to(device)
        return self

class DictObservationEncoder(nn.Module):
    def __init__(self, shape_meta: dict, num_frames: int):
        super().__init__()
        
        # Calculate lowdim_obs_dim from shape_meta
        lowdim_obs_dim = 0
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            if attr["type"] == "low_dim":
                lowdim_obs_dim += sum(attr["shape"])
        
        self.lowdim_obs_dim = lowdim_obs_dim

        low_dim_keys = list()
        key_shape_map = dict()
        key_transform_map = nn.ModuleDict()

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            obs_shape = tuple(attr["shape"])
            key_shape_map[key] = obs_shape

            obs_type = attr.get("type", "low_dim")
            if obs_type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        self.low_dim_keys = low_dim_keys
        self.num_frames = num_frames
        self.key_shape_map = key_shape_map

    def forward(self, obs_dict) -> torch.Tensor:

        low_dims = list()
        for key in self.low_dim_keys:
            low_dim = obs_dict[key].flatten(0, 1)
            assert low_dim.shape[1:] == self.key_shape_map[key]
            low_dims.append(low_dim)

        
        low_dims = torch.cat(low_dims, dim=-1)  # (B*T, D_low_dim)
        low_dims = rearrange(low_dims, "(b t) d -> b (t d)", t=self.num_frames)


        return low_dims

    @property
    def output_dim(self):
        return self.num_frames * sum(self.key_shape_map.values())
