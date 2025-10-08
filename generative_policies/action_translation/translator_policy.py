import torch
import torch.nn as nn
from typing import Union, Optional
import numpy as np

from generative_policies.action_translation.action_translator import ActionTranslatorInterface


class PolicyInterface(nn.Module):
    """
    Interface for a policy used in the ActionTranslatorPolicy

    SB3 policies satisfy this format
    """
    def __init__(self):
        super(PolicyInterface, self).__init__()
        
    def predict(self, obs, state, episode_start: Optional[np.ndarray] = None, deterministic: bool = True):
        """
        Predict an action given the observation, state, and episode start
        If deterministic is True, the action is deterministic. Otherwise, the action is stochastic
        - ex. for PPO, this is the mean action or the action sampled from the gaussian defined by the policy
        """
        raise NotImplementedError("Subclasses must implement predict method")


class ActionTranslatorPolicy:
    """
    Policy that combines a source policy and an action translator
    Runs the source policy, then translates the action using the action translator

    Designed for an SB3 policy
    """

    def __init__(self, source_policy, action_translator):
        self.source_policy: PolicyInterface = source_policy
        self.action_translator: ActionTranslatorInterface = action_translator

    def predict_base_and_translated(        
        self,
        policy_observation: Union[np.ndarray, dict[str, np.ndarray]],
        translator_observation: Optional[dict[str, np.ndarray]] = None,
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
        ):
        """
        Predict a base action and then translate it, returning both
        """
        base_prediction = self.source_policy.predict(policy_observation, state, episode_start, deterministic)
        # Extract just the action from the base policy prediction (which is a tuple of (action, state))
        base_action = base_prediction[0] if isinstance(base_prediction, tuple) else base_prediction
        translated_action = self.action_translator.predict(translator_observation, base_action)
        return translated_action, base_action