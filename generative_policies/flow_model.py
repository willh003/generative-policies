import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import NoisePredictionNet, UnetNoisePredictionNet
from .prior import GaussianPrior

class FlowModel(nn.Module):
    def __init__(self, noise_pred_net, num_train_steps=100, num_inference_steps=10, timeshift=1.0):
        super().__init__()
        self.noise_pred_net = noise_pred_net
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(0, 1, self.num_inference_steps + 1)
        self.timesteps = (timeshift * self.timesteps) / (1 + (timeshift - 1) * self.timesteps)

    @torch.no_grad()
    def sample(self, n_samples, device):
        # Initialize sample
        action = torch.randn(
            (n_samples, 1, self.noise_pred_net.input_dim), device=device
        )

        for tcont, tcont_next in zip(self.timesteps[:-1], self.timesteps[1:]):
            # Predict noise
            t = (tcont * self.num_train_steps).long()
            noise_pred = self.noise_pred_net(action, t)

            # Flow step
            action = action + (tcont_next - tcont) * noise_pred

        return action

    def forward(self, target):
        # Sample random noise
        noise = torch.randn_like(target)

        # Sample random timestep
        tcont = torch.rand((target.shape[0],), device=target.device)

        # Forward flow step
        direction = target - noise
        noisy_target = (
            noise + tcont.view(-1, *[1 for _ in range(target.dim() - 1)]) * direction
        )

        # Flow matching loss
        t = (tcont * self.num_train_steps).long()
        noise_pred = self.noise_pred_net(noisy_target, t)
        loss = F.mse_loss(noise_pred, direction)
        return loss

class ConditionalFlowModel(nn.Module):
    def __init__(
        self,
        noise_pred_net,
        num_train_steps=100,
        num_inference_steps=10,
        timeshift=1.0,
    ):
        super().__init__()

        # Noise prediction net
        self.noise_pred_net: NoisePredictionNet = noise_pred_net

        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        timesteps = torch.linspace(0, 1, self.num_inference_steps + 1)
        self.timesteps = (timeshift * timesteps) / (1 + (timeshift - 1) * timesteps)

    @torch.no_grad()
    def sample(self, global_cond, prior_sample):
        # Initialize sample
        action = prior_sample

        for tcont, tcont_next in zip(self.timesteps[:-1], self.timesteps[1:]):
            # Predict noise
            t = (tcont * self.num_train_steps).long()
            noise_pred = self.noise_pred_net(action, t, global_cond=global_cond)

            # Flow step
            action = action + (tcont_next - tcont) * noise_pred

        return action

    def forward(self, global_cond, prior_sample, action):

        # Sample random timestep
        tcont = torch.rand((action.shape[0],), device=action.device)

        # Forward flow step
        direction = action - prior_sample
        

        noisy_action = (
            prior_sample + tcont.view(-1, *[1 for _ in range(action.dim() - 1)]) * direction
        )

        # Flow matching loss
        t = (tcont * self.num_train_steps).long()
        noise_pred = self.noise_pred_net(noisy_action, t, global_cond=global_cond)
        loss = F.mse_loss(noise_pred, direction)
        return loss

class GaussianPriorFlowModel(ConditionalFlowModel):
    def __init__(self, 
                obs_encoder, 
                action_dim, 
                action_len, 
                num_train_steps=100, 
                num_inference_steps=10, 
                timeshift=1.0):
        """
        A flow policy that samples from a gaussian prior
        """
        
        noise_pred_net = UnetNoisePredictionNet(action_dim, global_cond_dim=obs_encoder.output_dim)

        super().__init__(noise_pred_net, num_train_steps, num_inference_steps, timeshift)
        self.obs_encoder = obs_encoder
        self.prior = GaussianPrior(action_dim, action_len)

    def sample(self, obs):

        prior_sample = self.prior.sample(n_samples=obs.shape[0], device=obs.device)
        global_cond = self.obs_encoder(obs)
        return super().sample(global_cond=global_cond, prior_sample=prior_sample)

    def forward(self, obs, action):
        prior_sample = self.prior.sample(n_samples=obs.shape[0], device=obs.device)
        global_cond = self.obs_encoder(obs)
        return super().forward(global_cond=global_cond, prior_sample=prior_sample, action=action)