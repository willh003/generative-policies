import torch
import torch.nn as nn

class GaussianPrior(nn.Module):
    def __init__(self, action_dim, action_len=None):
        """
        Args:
            action_dim: The dimension of the action.
            action_len: The length of the action. If None, the action is a single sample.
        """
        super().__init__()
        self.action_dim = action_dim
        self.action_len = action_len

    def sample(self, n_samples, device):
        """
        Sample from the prior. 
        Returns:
            action: shape (n_samples, action_len, action_dim) if action_len is provided, otherwise (n_samples, action_dim)

        """
        if self.action_len is None:
            action = torch.randn(
                (n_samples, self.action_dim), device=device
            )
        else:
            action = torch.randn(
                (n_samples, self.action_len, self.action_dim), device=device
            )
            
        return action