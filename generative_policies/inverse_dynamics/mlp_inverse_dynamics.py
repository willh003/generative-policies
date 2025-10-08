import torch
import torch.nn as nn
from typing import Union, Optional

from generative_policies.models.obs_encoder import IdentityEncoder, IdentityTwoInputEncoder

from generative_policies.inverse_dynamics.interface import InverseDynamicsInterface 

class MlpInverseDynamics(InverseDynamicsInterface):
    def __init__(self, action_dim, obs_dim, net_arch=[32, 32], device='cuda'):
        super(MlpInverseDynamics, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.net_arch = net_arch
        self.device = device

        self.cond_encoder = IdentityTwoInputEncoder(input_1_dim=obs_dim, input_2_dim=obs_dim, device=device)
        self.action_encoder = IdentityEncoder(action_dim, device=device)

        in_feat_dim = self.cond_encoder.output_dim

        layers = []
        prev_dim = in_feat_dim
        for hidden_units in net_arch:
            layers.append(nn.Linear(prev_dim, hidden_units))
            layers.append(nn.ReLU())
            prev_dim = hidden_units
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

        self.to(device)

    def forward(self, obs, next_obs, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For MlpActionTranslator, this computes MSE loss between predicted and target actions.
        """ 
        # Predict translated action given observation and action_prior
        cond = self.cond_encoder(obs, next_obs)
        action = self.action_encoder(action)

        sample = self.network(cond)
        # Compute MSE loss between predicted and target action
        loss = torch.nn.functional.mse_loss(sample, action)
        return loss
    
    def predict(self, obs, next_obs):
        """Predict translated action given observation and original action."""
        with torch.no_grad():
            cond = self.cond_encoder(obs, next_obs)
            action = self.network(cond)

        # Remove batch dimension if needed
        if len(obs.shape) == 1:
            action = action.squeeze(dim=0)  # type: ignore[assignment]

        return action.cpu().numpy()

    def to(self, device):
        self.device = device
        self.network.to(device)
        return self