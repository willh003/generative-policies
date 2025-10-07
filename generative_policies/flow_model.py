import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import ConditionalUNet1D


class ConditionalFlowModel(nn.Module):
    def __init__(self, target_dim, cond_dim=0, diffusion_step_embed_dim=32, down_dims=[32, 64, 128], source_sampler=None):
        super(ConditionalFlowModel, self).__init__()
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        # Default source sampler for when explicit prior samples/sampler are not provided per-call
        
        # Initialize the conditional UNet
        self.unet = ConditionalUNet1D(
            in_channels=target_dim,
            out_channels=target_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            cond_dim=cond_dim  # Only the actual condition dimension
        )
        
        # Source distribution sampler function. Defaults to gaussian
        if source_sampler is None:
            self.source_sampler = lambda batch_size, device: torch.randn(batch_size, target_dim, device=device)
        else:
            self.source_sampler = source_sampler

    def sample_source(self, batch_size, device, prior_samples=None, prior_sampler=None):
        """
        Choose the starting source samples for the flow integration.
        Priority (mutually exclusive intent):
          1) prior_samples: tensor of shape (batch_size, target_dim)
          2) prior_sampler: callable(batch_size, device) -> tensor
          3) fallback to self.source_sampler (which defaults to gaussian)
        """
        if prior_samples is not None:
            return prior_samples.to(device)
        if prior_sampler is not None:
            return prior_sampler(batch_size, device)
        return self.source_sampler(batch_size, device)

    def forward(self, target, condition=None, prior_samples=None, prior_sampler=None, device='cuda'):
        """
        target: (batch_size, target_dim)
        condition: (batch_size, cond_dim) or None (used only by network if cond_dim>0)
        prior_samples: Optional tensor (batch_size, target_dim) to use as sources
        prior_sampler: Optional callable to draw sources; ignored if prior_samples is provided
        Forward pass of the flow model, returning the loss from the target
        - Sample from the source distribution 
        - Find the true velocity field with target - source
        - Train the flow model to predict the velocity field
        """
        batch_size = target.shape[0]
        
        # Sample random time steps
        t = torch.rand(batch_size, 1, device=device)  # (batch_size, 1)
        
        # Sample starting sources (explicit prior overrides default sampler)
        source = self.sample_source(
            batch_size=batch_size,
            device=device,
            prior_samples=prior_samples,
            prior_sampler=prior_sampler,
        )  # (batch_size, target_dim)
        
        # Linear interpolation between source and target
        x_t = (1 - t) * source + t * target  # (batch_size, target_dim)
        
        # True velocity field (target - source)
        true_velocity = target - source  # (batch_size, target_dim)
        
        # Prepare condition for network: ignore provided condition if cond_dim == 0
        if self.cond_dim == 0:
            cond_input = torch.zeros(batch_size, 0, device=device)
        else:
            if condition is not None:
                # Ensure floating dtype for linear layers
                cond_input = condition
            else:
                cond_input = torch.zeros(batch_size, self.cond_dim, device=device)  # (batch_size, cond_dim)
        
        # Predict velocity field

        pred_velocity = self.unet(x_t, cond_input, t)  # (batch_size, target_dim)
        
        # Compute loss (MSE between predicted and true velocity)
        loss = F.mse_loss(pred_velocity, true_velocity)
        
        return loss
    
    def to(self, device):
        super().to(device)
        self.unet.to(device)
        return self
    
    def predict(self, batch_size, condition=None, prior_samples=None, prior_sampler=None, num_steps=100, device='cuda'):
        """
        Forward pass of the flow model, returning the predicted sample
        sample: (batch_size, target_dim)
        - Sample from the source distribution
        - Integrate the velocity field to get the predicted sample
        """
        self.eval()
        with torch.no_grad():
            # Start from source distribution (explicit prior overrides default sampler)
            x = self.sample_source(
                batch_size=batch_size,
                device=device,
                prior_samples=prior_samples,
                prior_sampler=prior_sampler,
            )  # (batch_size, target_dim)
            
            # Time steps for integration
            dt = 1.0 / num_steps
            
            for i in range(num_steps):
                t = torch.full((batch_size, 1), i * dt, device=device)  # (batch_size, 1)
                
                # Prepare condition for network: ignore provided condition if cond_dim == 0
                if self.cond_dim == 0:
                    cond_input = torch.zeros(batch_size, 0, device=device)
                else:
                    if condition is not None:
                        # Ensure floating dtype for linear layers
                        cond_input = condition
                    else:
                        cond_input = torch.zeros(batch_size, self.cond_dim, device=device)  # (batch_size, cond_dim)
                
                # Predict velocity
                velocity = self.unet(x, cond_input, t)  # (batch_size, target_dim)
                
                # Euler integration step
                x = x + velocity * dt
            
            return x



class LatentBridgeModel(ConditionalFlowModel):
    def __init__(self, target_dim, bridge_noise_sigma, cond_dim=0, diffusion_step_embed_dim=16, down_dims=[128, 256, 512], source_sampler=None):
        super(LatentBridgeModel, self).__init__(
            target_dim=target_dim,
            cond_dim=cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            source_sampler=source_sampler
        )
        self.bridge_noise_sigma = bridge_noise_sigma 

    def forward(self, target, condition=None, prior_samples=None, prior_sampler=None):
        """
        target: (batch_size, target_dim)
        condition: (batch_size, cond_dim) or None
        Forward pass of the LBM, returning the loss from the target
        - Sample from the source distribution (gaussian or uniform)
        - Find the stochastic velocity field with target - source + bridge noise
        - Train the flow model to predict the velocity field
        """
        batch_size = target.shape[0]
        device = target.device
        
        # Sample random time steps
        t = torch.rand(batch_size, 1, device=device)  # (batch_size, 1)
        
        # Sample from source distribution (explicit prior overrides default sampler)
        source = self.sample_source(
            batch_size=batch_size,
            device=device,
            prior_samples=prior_samples,
            prior_sampler=prior_sampler,
        )  # (batch_size, target_dim)
        

        # Linear interpolation between source and target
        bridge_noise = self.bridge_noise_sigma * torch.sqrt(t * (1-t)) * torch.randn(batch_size, self.target_dim, device=device)
        
        x_t = (1 - t) * source + t * target + bridge_noise  # (batch_size, target_dim)
        
        # True velocity field (target - source)
        true_velocity = target - source  # (batch_size, target_dim)
        
        # Prepare condition (separate from time)
        if self.cond_dim == 0:
            cond_input = torch.zeros(batch_size, 0, device=device)
        else:
            if condition is not None:
                cond_input = condition
            else:
                cond_input = torch.zeros(batch_size, self.cond_dim, device=device)
        
        # Predict velocity field
        pred_velocity = self.unet(x_t, cond_input, t)  # (batch_size, target_dim)
        
        # Compute loss (MSE between predicted and true velocity)
        loss = F.mse_loss(pred_velocity, true_velocity)
        
        return loss
    
    def to(self, device):
        super().to(device)
        self.unet.to(device)
        return self
    
    def predict(self, batch_size, condition=None, prior_samples=None, prior_sampler=None, num_steps=100, device='cpu'):
        """
        Forward pass of the flow model, returning the predicted sample
        sample: (batch_size, target_dim)
        - Sample from the source distribution
        - Integrate the velocity field to get the predicted sample
        """
        self.eval()
        with torch.no_grad():
            # Start from source distribution (explicit prior overrides default sampler)
            x = self.sample_source(
                batch_size=batch_size,
                device=device,
                prior_samples=prior_samples,
                prior_sampler=prior_sampler,
            )  # (batch_size, target_dim)
            
            # Time steps for integration
            dt = 1.0 / num_steps
            
            for i in range(num_steps):
                t = torch.full((batch_size, 1), i * dt, device=device)  # (batch_size, 1)
                
                # Prepare condition (separate from time)
                if self.cond_dim == 0:
                    cond_input = torch.zeros(batch_size, 0, device=device)
                else:
                    if condition is not None:
                        cond_input = condition
                    else:
                        cond_input = torch.zeros(batch_size, self.cond_dim, device=device)
                
                # Predict velocity
                velocity = self.unet(x, cond_input, t)  # (batch_size, target_dim)
                
                # Euler integration step
                x = x + velocity * dt

                bridge_noise = self.bridge_noise_sigma * torch.sqrt(t * (1-t)) * torch.randn(batch_size, self.target_dim, device=device)
                x = x + bridge_noise
            
            return x

