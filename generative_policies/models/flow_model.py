import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import ConditionalUNet1D
from .prior import GaussianPrior
from .transformer import DiT_S, DiT_B

class ConditionalFlowModel(nn.Module):
    def __init__(self, target_dim, cond_dim=0, diffusion_step_embed_dim=32, down_dims=[32, 64, 128], source_sampler=None, model_type='unet', use_spectral_norm=False):
        """
        use_spectral_norm: whether to use spectral normalization on the linear layers of the model (only for unet)
        """
        super(ConditionalFlowModel, self).__init__()
        self.target_dim = target_dim
        self.cond_dim = cond_dim

        assert model_type in ['unet', 'transformer'], "model_type must be either 'unet' or 'transformer'"
        if model_type == 'unet':
            self.model = ConditionalUNet1D(
                in_channels=target_dim,
                out_channels=target_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                cond_dim=cond_dim,  # Only the actual condition dimension
                use_spectral_norm=use_spectral_norm
            )
        elif model_type == 'transformer':   
            self.model = DiT_S(
                in_dim=target_dim,
                out_dim=target_dim,
                cond_dim=cond_dim,
            )
        
        # Source distribution sampler function. Defaults to gaussian
        if source_sampler is None:
            self.source_sampler = GaussianPrior(target_dim)
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
        self.train()
        batch_size = target.shape[0]
        if self.cond_dim == 0:
            cond_input = torch.zeros(batch_size, 0, device=device)
        else:
            if condition is not None:
                cond_input = condition
            else:
                cond_input = torch.zeros(batch_size, self.cond_dim, device=device)  # (batch_size, cond_dim)
        
        t = torch.rand(batch_size, 1, device=device)  # (batch_size, 1)
        source = self.sample_source(
            batch_size=batch_size,
            device=device,
            prior_samples=prior_samples,
            prior_sampler=prior_sampler,
        )  # (batch_size, target_dim)
        
        x_t = (1 - t) * source + t * target  # (batch_size, target_dim)
        pred_velocity = self.model(x_t, cond_input, t)  # (batch_size, target_dim)
        true_velocity = target - source  # (batch_size, target_dim)
        loss = F.mse_loss(pred_velocity, true_velocity)
        
        return loss
    
    def to(self, device):
        super().to(device)
        self.model.to(device)
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
            if self.cond_dim == 0:
                    cond_input = torch.zeros(batch_size, 0, device=device)
            else:
                if condition is not None:
                    cond_input = condition
                else:
                    cond_input = torch.zeros(batch_size, self.cond_dim, device=device)  # (batch_size, cond_dim)
            
            x = self.sample_source(
                batch_size=batch_size,
                device=device,
                prior_samples=prior_samples,
                prior_sampler=prior_sampler,
            )  # (batch_size, target_dim)
            
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t = torch.full((batch_size, 1), step * dt, device=device)  # (batch_size, 1)
                velocity = self.model(x, cond_input, t)  # (batch_size, target_dim)
                x = x + velocity * dt
            
            return x
        
    def compute_path_length(self, target, condition=None, prior_samples=None, prior_sampler=None, num_steps=100, device='cuda'):
        """
        Compute the path length of the flow model (approximate path integral of velocity field)
        """
            
        self.eval()
        batch_size = target.shape[0]
        with torch.no_grad():
            
            # Start from source distribution (explicit prior overrides default sampler)
            if self.cond_dim == 0:
                    cond_input = torch.zeros(batch_size, 0, device=device)
            else:
                if condition is not None:
                    cond_input = condition
                else:
                    cond_input = torch.zeros(batch_size, self.cond_dim, device=device)  # (batch_size, cond_dim)
            
            source = self.sample_source(
                batch_size=batch_size,
                device=device,
                prior_samples=prior_samples,
                prior_sampler=prior_sampler,
            )  # (batch_size, target_dim)

            x_t = source.clone()
            dt = 1.0 / num_steps
            integrated_path_length = 0.0

            for step in range(num_steps):
                t = torch.full((batch_size, 1), step * dt, device=device)  # (batch_size, 1)
                velocity = self.model(x_t, cond_input, t)
                dx = velocity * dt
                x_t += dx
                integrated_path_length += torch.norm(dx, dim=-1)
            final_sample = x_t
        straight_path_length = torch.norm(target - source, dim=-1)

        return integrated_path_length, straight_path_length, final_sample

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
        pred_velocity = self.model(x_t, cond_input, t)  # (batch_size, target_dim)
        
        # Compute loss (MSE between predicted and true velocity)
        loss = F.mse_loss(pred_velocity, true_velocity)
        
        return loss
    
    def to(self, device):
        super().to(device)
        self.model.to(device)
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
                velocity = self.model(x, cond_input, t)  # (batch_size, target_dim)
                
                # Euler integration step
                x = x + velocity * dt

                bridge_noise = self.bridge_noise_sigma * torch.sqrt(t * (1-t)) * torch.randn(batch_size, self.target_dim, device=device)
                x = x + bridge_noise
            
            return x




class EquilibriumMatchingModel(ConditionalFlowModel):
    def __init__(self, target_dim, cond_dim=0, 
            diffusion_step_embed_dim=16, down_dims=[128, 256, 512], 
            source_sampler=None, 
            lambda_=4.0, 
            gamma_type='truncated',
            mu=0.35,
            eta=0.003,
            g = .0001,
            objective_type='implicit',
            energy_type='dot',
            grad_type='nag',
            model_type='transformer'
        ):
        super(EquilibriumMatchingModel, self).__init__(
            target_dim=target_dim,
            cond_dim=cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            source_sampler=source_sampler,
            model_type=model_type
        )
        assert gamma_type in ['linear', 'truncated'], "gamma_type must be either 'linear' or 'truncated'"
        self.gamma_type = gamma_type
        self.lambda_ = lambda_
        self.g = g
        self.mu = mu
        self.eta = eta
        
        self.num_gamma = 100
        self.gamma_values = torch.linspace(0.0, 1.0, self.num_gamma)

        self.objective_type = objective_type
        self.energy_type = energy_type
        self.sample_fn = self.sample_gd if grad_type == 'gd' else self.sample_nag

        self.objective = self.implicit_objective if self.objective_type == 'implicit' else self.explicit_objective

    def forward(self, target, condition=None, prior_samples=None, prior_sampler=None):
        """
        The loss function takes as input model f, image x, and gradient
        magnitude function c, and returns the EqM loss.
        """
        batch_size = target.shape[0]
        device = target.device
        self.gamma_values = self.gamma_values.to(target.device)
        
        # Sample epsilon from source distribution (explicit prior overrides default sampler)
        eps = self.sample_source(
            batch_size=batch_size,
            device=device,
            prior_samples=prior_samples,
            prior_sampler=prior_sampler,
        )  # (batch_size, target_dim)
        
        # Prepare condition for network: ignore provided condition if cond_dim == 0
        if self.cond_dim == 0:
            cond_input = torch.zeros(batch_size, 0, device=device)
        else:
            if condition is not None:
                cond_input = condition
            else:
                cond_input = torch.zeros(batch_size, self.cond_dim, device=device)

        x = target.requires_grad_(True)

        gamma = torch.rand(batch_size, 1, device=device)
        xg = (1 - gamma) * eps + gamma * x  # (batch_size, target_dim)
        c_gamma = self.c_trunc(gamma) if self.gamma_type == 'truncated' else self.c_linear(gamma)
        t_mask = torch.zeros_like(gamma) # paper masks 
        pred = self.model(xg, cond_input, t_mask)  # (batch_size, target_dim)
        loss = self.objective(x, pred, eps, c_gamma)
        batch_loss = loss.mean()        

        return batch_loss

    def implicit_objective(self, x, pred, eps, c_gamma):
        objective = (pred - (eps - x) * c_gamma)**2
        return objective
    
    def explicit_objective(self, x, pred, eps, c_gamma, energy_type='dot'):
        
        def dot_product(x, pred):
            # Element-wise dot product for each sample in the batch
            return torch.sum(x * pred, dim=1, keepdim=True)
        
        def squared_l2_norm(x, pred):
            return torch.sum(-.5*torch.linalg.vector_norm(pred, dim=1, keepdim=True)**2)

        energy_function = dot_product if self.energy_type == 'dot' else squared_l2_norm
        g_x = energy_function(x, pred)

        grad_energy = torch.autograd.grad(g_x.sum(), x, retain_graph=True)[0]
        objective = (grad_energy - (eps - x) * c_gamma) ** 2
        return objective
        

    def c_linear(self, gamma):
        c = 1 - gamma
        return self.lambda_ * c

    def c_trunc(self,gamma, a=.8):
        c = torch.where(gamma <= a, 1, (1 - gamma) / (1 - a))
        return self.lambda_ * c

    def predict(self, batch_size, condition=None, prior_samples=None, prior_sampler=None, num_steps=1000, device='cpu', eta=None, mu=None, grad_type = None, g=None):

        """
        The sampling function takes in model f, initial state st, step size η,
        NAG factor µ, and threshold g, and returns the sample.
        def generate(f, st, eta, mu, g):
        x = st, x_last = st, grad = f(st)
        while norm(grad) < g:
        x_last = x
        x = x - eta*grad
        grad = f(x + mu*(x-x_last))
        return x
        """

        if eta is None:
            eta = self.eta
        if mu is None:
            mu = self.mu
        if g is None:
            g = self.g
        if grad_type is not None:
            self.sample_fn = self.sample_nag if grad_type == 'nag' else self.sample_gd

        self.eval()
        with torch.no_grad():
            # Start from source distribution (explicit prior overrides default sampler)
            x = self.sample_source(
                batch_size=batch_size,
                device=device,
                prior_samples=prior_samples,
                prior_sampler=prior_sampler,
            )  # (batch_size, target_dim)

            x_last = x.clone()
            
            # Prepare condition for network: ignore provided condition if cond_dim == 0
            if self.cond_dim == 0:
                cond_input = torch.zeros(batch_size, 0, device=device)
            else:
                if condition is not None:
                    cond_input = condition
                else:
                    cond_input = torch.zeros(batch_size, self.cond_dim, device=device)
            
            x = self.sample_fn(x, cond_input, num_steps, device)

        return x

    def sample_nag(self, x, cond_input, num_steps, device):
        batch_size = x.shape[0]
        t_mask = torch.zeros(batch_size, 1, device=device)
        grad = self.model(x, cond_input, t_mask)  # (batch_size, target_dim)
        for i in range(num_steps):
            grad_norm = torch.norm(grad, dim=1, keepdim=True) 
            if torch.all(grad_norm < self.g):
                print(f"Converged at step {i}")
                break

            x_last = x.clone()
            x = x - self.eta * grad
            nag_point = x + self.mu * (x - x_last)
            grad = self.model(nag_point, cond_input, t_mask)  # (batch_size, target_dim)
        return x

    def sample_gd(self, x, cond_input, num_steps, device):
        batch_size = x.shape[0]
        t_mask = torch.zeros(batch_size, 1, device=device)
        grad = self.model(x, cond_input, t_mask)  # (batch_size, target_dim)
        for i in range(num_steps):
            x = x - self.eta * grad
            grad = self.model(x, cond_input, t_mask)  # (batch_size, target_dim)
        return x