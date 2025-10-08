import torch
import torch.nn as nn

from generative_policies.models.flow_model import ConditionalFlowModel
from generative_policies.models.obs_encoder import IdentityEncoder, IdentityTwoInputEncoder
from generative_policies.models.prior import GaussianPrior

from generative_policies.action_translation.interface import ActionTranslatorInterface

class FlowActionPriorTranslator(ActionTranslatorInterface):
    """
    Action translator that learns p(a_trg | o), using a_src as prior.
    Does NOT condition explicitly on a_src
    """
    def __init__(self,
                action_dim, 
                obs_dim,
                diffusion_step_embed_dim=16,
                down_dims=[16, 32, 64],
                num_inference_steps=100,
                device='cuda'):
        super(FlowActionPriorTranslator, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.obs_encoder = IdentityEncoder(obs_dim, device=device)
        self.flow_model = ConditionalFlowModel(target_dim=action_dim, 
                                               cond_dim=self.obs_encoder.output_dim, 
                                               diffusion_step_embed_dim=diffusion_step_embed_dim,
                                               down_dims=down_dims,
                                               )
        self.num_inference_steps = num_inference_steps

        self.to(device)

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For FlowActionPriorTranslator, this would need to be implemented based on the flow policy's loss function.
        """        
        cond = self.obs_encoder(obs)
        action_prior = self.action_encoder(action_prior)
        action = self.action_encoder(action)
        loss = self.flow_model(target=action, condition=cond, prior_samples=action_prior, device=self.device)
        return loss

    def predict(self, obs, action_prior):
        cond = self.obs_encoder(obs)
        action_prior = self.action_encoder(action_prior)
        batch_size = obs.shape[0]
        sample = self.flow_model.predict(batch_size=batch_size, 
                                         condition=cond, 
                                         prior_samples=action_prior,
                                         num_steps=self.num_inference_steps, 
                                         device=self.device)
        return sample.cpu().numpy()

    def to(self, device):
        self.device = device
        self.flow_model.to(device)
        self.obs_encoder.to(device)
        return self


class FlowActionConditionedTranslator(ActionTranslatorInterface):
    """
    Action translator that learns p(a_trg | o, a_src), conditioned explicitly on a_src
    Uses N(0,1) as prior
    """
    def __init__(self,
                action_dim, 
                obs_dim,
                diffusion_step_embed_dim=16,
                down_dims=[16, 32, 64],
                num_inference_steps=100,
                device='cuda'):
        super(FlowActionConditionedTranslator, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.num_inference_steps = num_inference_steps

        self.cond_encoder = IdentityTwoInputEncoder(obs_dim, action_dim, device=device)
        self.action_encoder = IdentityEncoder(action_dim, device=device)
        
        action_prior = GaussianPrior(action_dim)
        self.flow_model = ConditionalFlowModel(target_dim=action_dim, 
                                               cond_dim=self.cond_encoder.output_dim, 
                                               diffusion_step_embed_dim=diffusion_step_embed_dim,
                                               down_dims=down_dims,
                                               source_sampler=action_prior,
                                               )

        self.to(device)

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For FlowActionPriorTranslator, this would need to be implemented based on the flow policy's loss function.
        """        
        cond = self.cond_encoder(obs, action_prior)
        action = self.action_encoder(action)
        loss = self.flow_model(target=action, condition=cond, device=self.device)
        return loss

    def predict(self, obs, action_prior):
        cond = self.cond_encoder(obs, action_prior)
        batch_size = cond.shape[0]
        sample = self.flow_model.predict(batch_size=batch_size, 
                                         condition=cond, 
                                         num_steps=self.num_inference_steps, 
                                         device=self.device)
        return sample.cpu().numpy()

    def to(self, device):
        self.device = device
        self.flow_model.to(device)
        self.cond_encoder.to(device)
        self.action_encoder.to(device)
        return self

    
class FlowActionPriorConditionedTranslator(FlowActionConditionedTranslator):
    """
    Acion translator that learns p(a_trg | o, a_src), conditioned explicitly and implicitly on a_src
    Uses a_src as the flow prior instead of N(0,1), and unet on a_src
    So basically this is FlowActionConditionedTranslator and FlowActionPriorTranslator combined
    """
    
    def __init__(self,
                action_dim, 
                obs_dim,
                diffusion_step_embed_dim=16,
                down_dims=[16, 32, 64],
                num_inference_steps=100,
                device='cuda'):
        super().__init__(action_dim, obs_dim, diffusion_step_embed_dim, down_dims, num_inference_steps, device)

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        For FlowActionPriorConditionedTranslator, this would need to be implemented based on the flow policy's loss function.
        """        
        cond = self.cond_encoder(obs, action_prior)
        action_prior = self.action_encoder(action_prior)
        action = self.action_encoder(action)
        loss = self.flow_model(target=action, condition=cond, prior_samples=action_prior, device=self.device)
        return loss

    def predict(self, obs, action_prior):
        cond = self.cond_encoder(obs, action_prior)
        action_prior = self.action_encoder(action_prior)
        batch_size = cond.shape[0]
        sample = self.flow_model.predict(batch_size=batch_size, 
                                         condition=cond, 
                                         prior_samples=action_prior,
                                         num_steps=self.num_inference_steps, 
                                         device=self.device)
        return sample.cpu().numpy()