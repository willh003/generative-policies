import torch
import torch.nn as nn

from generative_policies.models.flow_model import ConditionalFlowModel
from generative_policies.models.obs_encoder import IdentityEncoder, IdentityTwoInputEncoder, ImageEncoder, ImageStateEncoder
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
                device='cuda',
                model_type='unet',
                use_spectral_norm=False):
        super(FlowActionPriorTranslator, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.obs_encoder = IdentityEncoder(obs_dim, device=device)
        self.action_encoder = IdentityEncoder(action_dim, device=device)
        self.flow_model = ConditionalFlowModel(target_dim=action_dim, 
                                               cond_dim=self.obs_encoder.output_dim, 
                                               diffusion_step_embed_dim=diffusion_step_embed_dim,
                                               down_dims=down_dims,
                                               model_type=model_type,
                                               use_spectral_norm=use_spectral_norm)
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
        batch_size = cond.shape[0]

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
    
    def compute_path_length(self, obs, action_prior, action):
        cond = self.obs_encoder(obs)
        action_prior = self.action_encoder(action_prior)
        action = self.action_encoder(action)

        integrated_path_length, straight_path_length, final_sample = self.flow_model.compute_path_length(
                                         target=action,
                                         condition=cond,
                                         prior_samples=action_prior,
                                         num_steps=self.num_inference_steps,
                                         device=self.device)
        
        return integrated_path_length, straight_path_length, final_sample.cpu().numpy()

class FlowImageActionPriorTranslator(FlowActionPriorTranslator):
    def __init__(self, action_dim, image_dim, diffusion_step_embed_dim=16, down_dims=[16, 32, 64], num_inference_steps=100, device='cuda', model_type='unet', use_spectral_norm=False):
        
        super(FlowImageActionPriorTranslator, self).__init__(action_dim, image_dim, diffusion_step_embed_dim, down_dims, num_inference_steps, device, model_type, use_spectral_norm)
        
        self.obs_encoder = ImageEncoder(image_dim, device=device)
        self.flow_model = ConditionalFlowModel(target_dim=action_dim, 
                                               cond_dim=self.obs_encoder.output_dim, 
                                               diffusion_step_embed_dim=diffusion_step_embed_dim,
                                               down_dims=down_dims,
                                               model_type=model_type,
                                               use_spectral_norm=use_spectral_norm)
        self.to(device)

class FlowImageStateActionPriorTranslator(FlowActionPriorTranslator):
    def __init__(self, action_dim, image_dim, state_dim, diffusion_step_embed_dim=16, down_dims=[16, 32, 64], num_inference_steps=100, device='cuda', model_type='unet', use_spectral_norm=False):
        super(FlowImageStateActionPriorTranslator, self).__init__(action_dim, state_dim, diffusion_step_embed_dim, down_dims, num_inference_steps, device, model_type, use_spectral_norm)
        
        self.obs_encoder = ImageStateEncoder(image_dim, state_dim, device=device) # image_dim + state_dim -> obs_dim
        self.flow_model = ConditionalFlowModel(target_dim=action_dim, 
                                               cond_dim=self.obs_encoder.output_dim, 
                                               diffusion_step_embed_dim=diffusion_step_embed_dim,
                                               down_dims=down_dims,
                                               model_type=model_type,
                                               use_spectral_norm=use_spectral_norm)
        self.to(device)


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
                device='cuda',
                model_type='unet',
                use_spectral_norm=False):
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
                                               model_type=model_type,
                                               use_spectral_norm=use_spectral_norm)

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

    def compute_path_length(self, obs, action_prior, action):
        cond = self.cond_encoder(obs, action_prior)
        action = self.action_encoder(action)
        integrated_path_length, straight_path_length, final_sample = self.flow_model.compute_path_length(
                                         target=action,
                                         condition=cond,
                                         num_steps=self.num_inference_steps,
                                         device=self.device)
        
        return integrated_path_length, straight_path_length, final_sample.cpu().numpy()



class FlowDeltaTranslator(FlowActionConditionedTranslator):
    """
    Action translator that learns p(a_trg | o, a_src), conditioned explicitly on a_src
    Predicts the delta a_trg - a_src instead of the action itself, using 
    N(0,1) as the prior for the delta
    """
    
    def __init__(self,
                action_dim, 
                obs_dim,
                diffusion_step_embed_dim=16,
                down_dims=[16, 32, 64],
                num_inference_steps=100,
                device='cuda',
                model_type='unet'):
        super(FlowDeltaTranslator, self).__init__(action_dim, obs_dim, diffusion_step_embed_dim, down_dims, num_inference_steps, device, model_type)

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        """        
        cond = self.cond_encoder(obs, action_prior)
        action_prior = self.action_encoder(action_prior)
        action = self.action_encoder(action)
        action_delta = action - action_prior
        loss = self.flow_model(target=action_delta, condition=cond, device=self.device)
        
        return loss

    def predict(self, obs, action_prior):
        cond = self.cond_encoder(obs, action_prior)
        action_prior = self.action_encoder(action_prior)
        batch_size = cond.shape[0]
        sample = self.flow_model.predict(batch_size=batch_size, 
                                         condition=cond, 
                                         num_steps=self.num_inference_steps, 
                                         device=self.device)
        return sample.cpu().numpy() + action_prior.cpu().numpy()

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
                device='cuda',
                model_type='unet'):
        super().__init__(action_dim, obs_dim, diffusion_step_embed_dim, down_dims, num_inference_steps, device, model_type)

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


    def compute_path_length(self, obs, action_prior, action):
        cond = self.cond_encoder(obs, action_prior)
        action_prior = self.action_encoder(action_prior)
        action = self.action_encoder(action)
        integrated_path_length, straight_path_length, final_sample = self.flow_model.compute_path_length(target=action, condition=cond, prior_samples=action_prior, device=self.device)
        
        return integrated_path_length, straight_path_length, final_sample.cpu().numpy()

    

class FlowBC(ActionTranslatorInterface):
    """
    BC policy that learns p(a | o) by flowing from N(0,1) to a, conditioned on o
    I.e., no action conditioning or prior
    """
    def __init__(self,
                action_dim, 
                obs_dim,
                diffusion_step_embed_dim=16,
                down_dims=[16, 32, 64],
                num_inference_steps=100,
                device='cuda',
                model_type='unet',
                use_spectral_norm=False):
        super(FlowBC, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.num_inference_steps = num_inference_steps

        self.cond_encoder = IdentityEncoder(obs_dim, device=device)
        self.action_encoder = IdentityEncoder(action_dim, device=device)
        
        action_prior = GaussianPrior(action_dim)
        self.flow_model = ConditionalFlowModel(target_dim=action_dim, 
                                               cond_dim=self.cond_encoder.output_dim, 
                                               diffusion_step_embed_dim=diffusion_step_embed_dim,
                                               down_dims=down_dims,
                                               source_sampler=action_prior,
                                               model_type=model_type,
                                               use_spectral_norm=use_spectral_norm)

        self.to(device)

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of observations, action_priors, and actions.
        NOTE: ignores the action_prior in this class, since this is action-unconditional
        """        
        cond = self.cond_encoder(obs)
        action = self.action_encoder(action)
        loss = self.flow_model(target=action, condition=cond, device=self.device)
        return loss

    def predict(self, obs, action_prior):
        """
        NOTE: ignores the action_prior in this class
        """
        cond = self.cond_encoder(obs)
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

    def compute_path_length(self, obs, action_prior, action):
        cond = self.cond_encoder(obs)
        action = self.action_encoder(action)
        integrated_path_length, straight_path_length, final_sample = self.flow_model.compute_path_length(target=action, condition=cond, device=self.device)
        
        return integrated_path_length, straight_path_length, final_sample.cpu().numpy()



class FlowActionOnly(ActionTranslatorInterface):
    """
    Flow policy that learns p(a_trg | a_src) by flowing from a_src to a_trg, with conditioning on a_src
    I.e., ignores the observation
    """
    def __init__(self,
                action_dim, 
                diffusion_step_embed_dim=16,
                down_dims=[16, 32, 64],
                num_inference_steps=100,
                device='cuda',
                model_type='unet'):
        super(FlowActionOnly, self).__init__()
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps

        self.action_encoder = IdentityEncoder(action_dim, device=device)
        
        action_prior = GaussianPrior(action_dim)
        self.flow_model = ConditionalFlowModel(target_dim=action_dim, 
                                               cond_dim=self.action_encoder.output_dim, 
                                               diffusion_step_embed_dim=diffusion_step_embed_dim,
                                               down_dims=down_dims,
                                               source_sampler=action_prior,
                                               model_type=model_type,
                                               use_spectral_norm=use_spectral_norm)

        self.to(device)

    def forward(self, obs, action_prior, action) -> torch.Tensor:
        """
        Compute the loss for a sampled batch of action_priors, and actions.
        NOTE: ignores the obs in this class
        """        
        cond = self.action_encoder(action)
        action = self.action_encoder(action)
        loss = self.flow_model(target=action, condition=cond, prior_samples=action_prior, device=self.device)
        return loss

    def predict(self, obs, action_prior):
        """
        NOTE: ignores the obs in this class
        """
        action_prior = self.action_encoder(action_prior)
        cond = self.action_encoder(action_prior)
        batch_size = cond.shape[0]

        sample = self.flow_model.predict(batch_size=batch_size, 
                                         condition=cond, 
                                         prior_samples=action_prior,
                                         num_steps=self.num_inference_steps, 
                                         device=self.device)
        return sample.cpu().numpy()

    def to(self, device):
        self.device = device
        self.flow_model.to(device)
        self.action_encoder.to(device)
        return self