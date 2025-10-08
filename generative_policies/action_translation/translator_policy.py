import torch
import torch.nn as nn
from typing import Union, Optional
import numpy as np

from generative_policies.action_translation.interface import ActionTranslatorInterface, PolicyInterface

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