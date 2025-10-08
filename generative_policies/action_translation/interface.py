import torch
import torch.nn as nn
from typing import Optional
import numpy as np

class ActionTranslatorInterface(nn.Module):
    """
    An action translator that translates an action from one domain to another, conditioned on an observation
    """

    def __init__(self):
        super(ActionTranslatorInterface, self).__init__()
    
    def predict(self, obs, action_prior) -> np.ndarray:
        """
        Predict an action given the observation and the action_prior
        Returns the action as a numpy array, for ease of env stepping
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def to(self, device):
        """
        Move the model to the specified device
        """
        raise NotImplementedError("Subclasses must implement to method")

class PolicyInterface(nn.Module):
    """
    Interface for a policy used in the ActionTranslatorPolicy

    SB3 policies satisfy this format
    """

    def __init__(self):
        super(PolicyInterface, self).__init__()
        
    def predict(self, 
                obs,  
                state: Optional[tuple[np.ndarray, ...]] = None, 
                episode_start: Optional[np.ndarray] = None, 
                deterministic: bool = True
            ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Predict an action given the observation, state, and episode start
        If deterministic is True, the action is deterministic. Otherwise, the action is stochastic
        - ex. for PPO, this is the mean action or the action sampled from the gaussian defined by the policy
        """
        raise NotImplementedError("Subclasses must implement predict method")
