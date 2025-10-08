import torch
import torch.nn as nn

class GaussianPrior(nn.Module):
    def __init__(self, prior_dim):
        """
        Args:
            prior_dim: The dimension of the prior (source distribution).
        """
        super().__init__()
        self.prior_dim = prior_dim

    def forward(self, batch_size, device):
        """
        Sample from the prior. 
        Returns:
            prior: shape (n_samples, prior_dim)

        """
        prior = torch.randn(
            batch_size, self.prior_dim, device=device
        )
            
        return prior