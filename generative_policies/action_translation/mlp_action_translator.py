
import torch
import torch.nn as nn

from generative_policies.models.obs_encoder import IdentityEncoder, IdentityTwoInputEncoder
from generative_policies.action_translation.interface import ActionTranslatorInterface


class MlpActionTranslator(ActionTranslatorInterface):
    def __init__(self, action_dim, obs_dim, device='cuda', net_arch=None):
        super(MlpActionTranslator, self).__init__()
        self.action_dim = action_dim
        self.cond_encoder = IdentityTwoInputEncoder(input_1_dim=obs_dim, input_2_dim=action_dim, device=device)

        self.action_encoder = IdentityEncoder(action_dim, device=device)

        in_feat_dim = self.cond_encoder.output_dim
        # Default to two hidden layers of 32 units each for backward compatibility
        if net_arch is None:
            net_arch = [32, 32]

        layers = []
        prev_dim = in_feat_dim
        for hidden_units in net_arch:
            layers.append(nn.Linear(prev_dim, hidden_units))
            layers.append(nn.ReLU())
            prev_dim = hidden_units
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

        self.to(device)

    def to(self, device):
        self.device = device
        self.network.to(device)
        return self

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For MlpActionTranslator, this computes MSE loss between predicted and target actions.
        """ 
        # Predict translated action given observation and action_prior
        cond = self.cond_encoder(obs, action_prior)
        action = self.action_encoder(action)

        predicted_action = self.network(cond)
        # Compute MSE loss between predicted and target action
        if len(action.shape) == 3:
            action = action.squeeze(dim=1)
        loss = torch.nn.functional.mse_loss(predicted_action, action)
        return loss
    
    def compute_path_length(self, obs, action_prior, action):
        """
        For compatibility with FlowActionTranslator. Returns zeros for path length and predicted action.
        """
        cond = self.cond_encoder(obs, action_prior)
        predicted_action = self.network(cond)

        batch_size = action.shape[0]
        return torch.zeros(batch_size), torch.zeros(batch_size), predicted_action.cpu().numpy()
        
    def is_vectorized_observation(self, obs):
        return len(obs.shape) > 1
    
    def predict(self, obs, action_prior):
        """Predict translated action given observation and original action."""
        with torch.no_grad():
            cond = self.cond_encoder(obs, action_prior)
            translated_action = self.network(cond)

        # Remove batch dimension if needed
        if not self.is_vectorized_observation(obs):
            translated_action = translated_action.squeeze(dim=0)  # type: ignore[assignment]
        return translated_action.cpu().numpy()

