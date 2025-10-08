import torch
import torch.nn as nn
import numpy as np

class InverseDynamicsInterface(nn.Module):
    """
    Interface for an inverse dynamics model
    """

    def __init__(self):
        super(InverseDynamicsInterface, self).__init__()
        
    def predict(self, obs, next_obs) -> np.ndarray:
        """
        Predict an action given the observation and the next observation
        Returns the action as a numpy array, for ease of env stepping
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def forward(self, obs, next_obs, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, next observations, and actions
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def to(self, device):
        """
        Move the model to the specified device
        """
        raise NotImplementedError("Subclasses must implement to method")
