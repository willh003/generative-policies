# Generative Policies

A lightweight Python library implementing generative models for robot learning applications. Provides simple interfaces for flow-based models, as well as model explicitly designed for inverse dynamics and action generation (policies).

## Overview

This library provides implementations of conditional flow models and related components for robotics applications, including:

- **Flow Models**: Conditional flow models for learning complex distributions
- **Inverse Dynamics**: Models for learning action distributions from state transitions
- **Action Translation**: Tools for translating actions between different domains
- **Observation Encoders**: Utilities for handling various observation formats

## Installation

```bash
conda create -n generative_policies
conda activate generative_policies
conda install pip
pip install -e .
```

### Dependencies

- python >= 3.8
- torch
- einops
- numpy

## Example

Run flow matching on bimodal data (see experiment.py for other experiments)
```
python -m generative_policies.models.experiment
```

## Core Components

### Flow Models

#### `ConditionalFlowModel`
A conditional flow model that learns to transform from a source distribution to a target distribution.

```python
from generative_policies.models import ConditionalFlowModel

model = ConditionalFlowModel(
    target_dim=action_dim,
    cond_dim=observation_dim,
    diffusion_step_embed_dim=32,
    down_dims=[32, 64, 128]
)

# Training
loss = model(target_actions, condition=observations)

# Inference
predicted_actions = model.predict(
    batch_size=32, 
    condition=observations,
    num_steps=100
)
```

#### `LatentBridgeModel`
A specialized flow model with bridge noise for more stable training. In practice, doesn't seem to work as well as flow model.

### Inverse Dynamics

#### `MlpInverseDynamics`
MLP-based inverse dynamics model for learning action distributions from state transitions.

```python
from generative_policies.inverse_dynamics import MlpInverseDynamics

model = MlpInverseDynamics(
    action_dim=7,
    obs_dim=64,
    net_arch=[32, 32]
)

# Training
loss = model(obs, next_obs, actions)

# Prediction
predicted_actions = model.predict(obs, next_obs)
```

#### `FlowInverseDynamics`
Flow-based inverse dynamics model using conditional flow models.

### Action Translation

#### `MlpActionTranslator`
MLP-based action translator for converting actions between domains.

```python
from generative_policies.action_translation import MlpActionTranslator

translator = MlpActionTranslator(
    action_dim=7,
    obs_dim=64,
    net_arch=[32, 32]
)

# Training
loss = translator(obs, source_actions, target_actions)

# Translation
translated_actions = translator.predict(obs, source_actions)
```

#### Flow-based Translators
- `FlowActionPriorTranslator`: Uses source action as prior
- `FlowActionConditionedTranslator`: Conditions on source action, uses gaussian as prior
- `FlowActionPriorConditionedTranslator`: Uses source action as prior AND for explicit conditioning

### Observation Encoders

#### `IdentityEncoder`
Handles simple tensor/array observations with automatic device management.

#### `IdentityTwoInputEncoder`
Encodes two input tensors into a single tensor. Useful for conditioning on (s,s') or (s,a)

#### `DictObservationEncoder`
Handles dictionary-based observations with shape metadata.

### Utilities

#### `GaussianPrior`
Simple Gaussian prior distribution for flow models.

#### `ConditionalUNet1D`
1D conditional U-Net architecture for flow models.

## Usage Examples

### Basic Flow Model Training

```python
import torch
from generative_policies.models import ConditionalFlowModel

# Initialize model
model = ConditionalFlowModel(target_dim=7, cond_dim=64)

# Training loop
optimizer = torch.optim.Adam(model.parameters())
for batch in dataloader:
    actions, observations = batch
    loss = model(actions, condition=observations)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Action Translation

```python
from generative_policies.action_translation import MlpActionTranslator

# Train translator
translator = MlpActionTranslator(action_dim=7, obs_dim=64)
for batch in dataloader:
    obs, source_actions, target_actions = batch
    loss = translator(obs, source_actions, target_actions)
    # ... training step

# Use translator
translated = translator.predict(observation, source_action)
```

## API Reference

### Models
- `ConditionalFlowModel`: Main flow model implementation
- `LatentBridgeModel`: Flow model with bridge noise
- `GaussianPrior`: Gaussian prior distribution
- `ConditionalUNet1D`: U-Net architecture

### Inverse Dynamics
- `MlpInverseDynamics`: MLP-based inverse dynamics
- `FlowInverseDynamics`: Flow-based inverse dynamics

### Action Translation
- `MlpActionTranslator`: MLP action translator
- `FlowActionPriorTranslator`: Flow translator using source as prior
- `FlowActionConditionedTranslator`: Flow translator conditioning on source
- `FlowActionPriorConditionedTranslator`: Combined approach

### Encoders
- `IdentityEncoder`: Simple tensor encoder
- `IdentityTwoInputEncoder`: Two-input encoder
- `DictObservationEncoder`: Dictionary observation encoder

## License

MIT License
