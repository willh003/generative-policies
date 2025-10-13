
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm


def train_flow_model(flow_model, train_data, val_data, epochs, batch_size, learning_rate, device='cpu', verbose=True, train_conditions=None, val_conditions=None, train_prior_samples=None, val_prior_samples=None, train_prior_sampler=None, val_prior_sampler=None):
    """
    Train the flow model using flow matching
    
    Args:
        flow_model: FlowMatchingModel instance
        train_data: Training data tensor (N, target_dim)
        val_data: Validation data tensor (M, target_dim)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        verbose: Whether to print training progress
        train_conditions: Training conditions tensor (N, cond_dim) or None
        val_conditions: Validation conditions tensor (M, cond_dim) or None
    
    Returns:
        dict: Training history with losses
    """
    # Move model to device
    flow_model = flow_model.to(device)
    
    # Create data loaders
    features = [train_data]
    val_features = [val_data]
    if train_conditions is not None:
        features.append(train_conditions)
        val_features.append(val_conditions)
    if train_prior_samples is not None:
        features.append(train_prior_samples)
        val_features.append(val_prior_samples)
    train_dataset = TensorDataset(*features)
    val_dataset = TensorDataset(*val_features)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize optimizer
    optimizer = optim.Adam(flow_model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        flow_model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_data in train_loader:
            idx = 0
            target = batch_data[idx].to(device); idx += 1
            condition = None
            prior_samples = None
            if train_conditions is not None:
                condition = batch_data[idx].to(device); idx += 1
            if train_prior_samples is not None:
                prior_samples = batch_data[idx].to(device); idx += 1
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = flow_model(target, condition, prior_samples=prior_samples, prior_sampler=train_prior_sampler)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        flow_model.eval()
        val_loss = 0.0
        val_batches = 0
        
        #with torch.no_grad(): # No grad breaks explicit energy computation
        for batch_data in val_loader:
            idx = 0
            target = batch_data[idx].to(device); idx += 1
            condition = None
            prior_samples = None
            if val_conditions is not None:
                condition = batch_data[idx].to(device); idx += 1
            if val_prior_samples is not None:
                prior_samples = batch_data[idx].to(device); idx += 1
            
            # Forward pass
            loss = flow_model(target, condition, prior_samples=prior_samples, prior_sampler=val_prior_sampler)
            
            val_loss += loss.item()
            val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return history


def evaluate_flow_model(flow_model, test_data, num_samples=1000, device='cpu', eta=None, mu=None, grad_type=None):
    """
    Evaluate the flow model by generating samples and computing metrics
    
    Args:
        flow_model: Trained FlowMatchingModel instance
        test_data: Test data tensor (N, target_dim)
        num_samples: Number of samples to generate for evaluation
        device: Device to run evaluation on
        eta: Optional eta parameter for EquilibriumMatchingModel inference
        mu: Optional mu parameter for EquilibriumMatchingModel inference
        grad_type: Optional grad_type parameter for EquilibriumMatchingModel inference
    
    Returns:
        dict: Evaluation metrics
    """
    flow_model = flow_model.to(device)
    flow_model.eval()
    
    # Generate samples
    with torch.no_grad():
        if eta is not None and mu is not None:
            # For EquilibriumMatchingModel, pass eta, mu, and grad_type parameters
            generated_samples = flow_model.predict(
                batch_size=num_samples, 
                device=device,
                eta=eta,
                mu=mu,
                grad_type=grad_type
            ).cpu().numpy()
        else:
            # For other models, use default parameters
            generated_samples = flow_model.predict(
                batch_size=num_samples, 
                device=device
            ).cpu().numpy()
    
    test_data_np = test_data.cpu().numpy()
    
    # Compute basic statistics
    metrics = {}
    
    # Mean and std comparison
    metrics['test_mean'] = np.mean(test_data_np, axis=0)
    metrics['test_std'] = np.std(test_data_np, axis=0)
    metrics['generated_mean'] = np.mean(generated_samples, axis=0)
    metrics['generated_std'] = np.std(generated_samples, axis=0)
    
    # Mean squared error between test and generated statistics
    metrics['mean_mse'] = np.mean((metrics['test_mean'] - metrics['generated_mean']) ** 2)
    metrics['std_mse'] = np.mean((metrics['test_std'] - metrics['generated_std']) ** 2)
    
    # Wasserstein distance (approximation using 1D case for each dimension)
    from scipy.stats import wasserstein_distance
    wasserstein_distances = []
    for i in range(test_data_np.shape[1]):
        wd = wasserstein_distance(test_data_np[:, i], generated_samples[:, i])
        wasserstein_distances.append(wd)
    
    metrics['mean_wasserstein'] = np.mean(wasserstein_distances)
    metrics['max_wasserstein'] = np.max(wasserstein_distances)
    
    return metrics


def create_data_loader(data, batch_size, shuffle=True):
    """
    Helper function to create a DataLoader from data tensor
    
    Args:
        data: Data tensor (N, target_dim)
        batch_size: Batch size
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
