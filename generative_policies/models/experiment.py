
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from .flow_model import ConditionalFlowModel, LatentBridgeModel, EquilibriumMatchingModel
from .train import train_flow_model, evaluate_flow_model
from cluster_utils import set_cluster_graphics_vars
set_cluster_graphics_vars()
def gaussian_sampler(target_dim, mean=0.0, std=1.0):
    """Create a Gaussian source sampler function"""
    def sampler(batch_size, device):
        return torch.randn(batch_size, target_dim, device=device) * std + mean
    return sampler


def uniform_sampler(target_dim, min_val=-3.0, max_val=3.0):
    """Create a uniform source sampler function"""
    def sampler(batch_size, device):
        return torch.rand(batch_size, target_dim, device=device) * (max_val - min_val) + min_val
    return sampler


def mixture_sampler(target_dim, means=None, stds=None, weights=None):
    """Create a mixture of Gaussians source sampler function"""
    if means is None:
        means = torch.tensor([[-2.0, -2.0], [2.0, 2.0], [0.0, 0.0]]).T  # (target_dim, num_components)
    if stds is None:
        stds = torch.tensor([[0.5, 0.5], [0.5, 0.5], [1.0, 1.0]]).T  # (target_dim, num_components)
    if weights is None:
        weights = torch.tensor([0.4, 0.4, 0.2])  # (num_components,)
    
    def sampler(batch_size, device):
        # Move parameters to device (create new variables to avoid scoping issues)
        means_device = means.to(device)
        stds_device = stds.to(device)
        weights_device = weights.to(device)
        
        # Choose components based on weights
        component_indices = torch.multinomial(weights_device, batch_size, replacement=True)
        samples = torch.zeros(batch_size, target_dim, device=device)
        for i in range(batch_size):
            comp_idx = component_indices[i]
            mean = means_device[:, comp_idx]
            std = stds_device[:, comp_idx]
            samples[i] = torch.randn(target_dim, device=device) * std + mean
        return samples
    return sampler


def ring_sampler(target_dim, center=None, radius=2.0, width=0.5):
    """Create a ring source sampler function"""
    if center is None:
        center = torch.zeros(target_dim)
    elif isinstance(center, (tuple, list)):
        center = torch.tensor(center, dtype=torch.float32)
    
    def sampler(batch_size, device):
        center_device = center.to(device)
        # Sample angle uniformly
        theta = torch.rand(batch_size, device=device) * 2 * math.pi
        # Sample radius from normal around radius
        r = torch.randn(batch_size, device=device) * width + radius
        # Convert to Cartesian
        if target_dim == 2:
            x = r * torch.cos(theta) + center_device[0]
            y = r * torch.sin(theta) + center_device[1]
            return torch.stack([x, y], dim=1)
        else:
            # For higher dimensions, use spherical coordinates
            samples = torch.zeros(batch_size, target_dim, device=device)
            for i in range(batch_size):
                # Sample direction uniformly on sphere
                direction = torch.randn(target_dim, device=device)
                direction = direction / torch.norm(direction)
                samples[i] = center_device + r[i] * direction
            return samples
    return sampler


def spiral_sampler(target_dim, center=None, turns=2.0, radius_max=2.0, noise=0.1):
    """Create a spiral source sampler function"""
    if center is None:
        center = torch.zeros(target_dim)
    elif isinstance(center, (tuple, list)):
        center = torch.tensor(center, dtype=torch.float32)
    
    def sampler(batch_size, device):
        center_device = center.to(device)
        # Sample parameter t uniformly from 0 to 1
        t = torch.rand(batch_size, device=device)
        # Convert to angle (0 to turns * 2π)
        theta = t * turns * 2 * math.pi
        # Radius grows linearly with t
        r = t * radius_max
        # Add noise
        r_noise = r + torch.randn(batch_size, device=device) * noise
        # Convert to Cartesian
        if target_dim == 2:
            x = r_noise * torch.cos(theta) + center_device[0]
            y = r_noise * torch.sin(theta) + center_device[1]
            return torch.stack([x, y], dim=1)
        else:
            # For higher dimensions, use the first two dimensions for spiral
            samples = torch.zeros(batch_size, target_dim, device=device)
            samples[:, 0] = r_noise * torch.cos(theta) + center_device[0]
            samples[:, 1] = r_noise * torch.sin(theta) + center_device[1]
            # Fill remaining dimensions with noise
            if target_dim > 2:
                samples[:, 2:] = torch.randn(batch_size, target_dim - 2, device=device) * 0.1
            return samples
    return sampler


def generate_bimodal_data(n_samples=5000, noise=0.1):
    """
    Generate a bimodal 2D distribution for the experiment
    
    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of noise to add
    
    Returns:
        torch.Tensor: Generated data of shape (n_samples, 2)
    """
    # Two Gaussian modes
    mode1 = np.random.multivariate_normal([2, 2], [[0.5, 0.2], [0.2, 0.5]], n_samples // 2)
    mode2 = np.random.multivariate_normal([-2, -2], [[0.5, -0.2], [-0.2, 0.5]], n_samples // 2)
    
    # Combine modes
    data = np.vstack([mode1, mode2])
    
    # Add noise
    data += np.random.normal(0, noise, data.shape)
    
    # Shuffle
    np.random.shuffle(data)
    
    return torch.tensor(data, dtype=torch.float32)


def plot_data_comparison(target_data, generated_data, save_path=None, title=None):
    """
    Plot comparison between target and generated data
    
    Args:
        target_data: Target data tensor (N, 2)
        generated_data: Generated data tensor (N, 2)
        save_path: Optional path to save the plot
        title: Optional title for the overall plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Convert to numpy for plotting
    target_np = target_data.detach().cpu().numpy()
    generated_np = generated_data.detach().cpu().numpy()
    
    # Target data
    axes[0].scatter(target_np[:, 0], target_np[:, 1], alpha=0.6, s=1)
    axes[0].set_title('Target Data (Bimodal)')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].grid(True, alpha=0.3)
    
    # Generated data
    axes[1].scatter(generated_np[:, 0], generated_np[:, 1], alpha=0.6, s=1, color='red')
    axes[1].set_title('Generated Data')
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].grid(True, alpha=0.3)
    
    # Overlay comparison
    axes[2].scatter(target_np[:, 0], target_np[:, 1], alpha=0.4, s=1, label='Target', color='blue')
    axes[2].scatter(generated_np[:, 0], generated_np[:, 1], alpha=0.4, s=1, label='Generated', color='red')
    axes[2].set_title('Overlay Comparison')
    axes[2].set_xlabel('X1')
    axes[2].set_ylabel('X2')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_spiral_target_data(n_samples=10000, center=(0, 0), turns=3.0, radius_max=3.0, noise=0.1):
    """
    Generate spiral target data with specific parameters
    
    Args:
        n_samples: Number of samples to generate
        center: Center of the spiral (x, y)
        turns: Number of turns in the spiral
        radius_max: Maximum radius of the spiral
        noise: Standard deviation of noise to add
    
    Returns:
        torch.Tensor: Generated data of shape (n_samples, 2)
    """
    # Generate parameter t uniformly from 0 to 1
    t = np.linspace(0, 1, n_samples)
    # Add some randomness to make it more realistic
    t += np.random.normal(0, 0.02, n_samples)
    t = np.clip(t, 0, 1)
    
    # Convert to angle (0 to turns * 2π)
    theta = t * turns * 2 * np.pi
    # Radius grows linearly with t
    r = t * radius_max
    
    # Convert to Cartesian
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    
    # Add noise
    x += np.random.normal(0, noise, n_samples)
    y += np.random.normal(0, noise, n_samples)
    
    data = np.column_stack([x, y])
    return torch.tensor(data, dtype=torch.float32)


def generate_conditional_data(n_samples_per_class=2500, noise=0.1):
    """
    Generate conditional 2D data with complex distributions based on condition
    
    Args:
        n_samples_per_class: Number of samples per condition class
        noise: Standard deviation of noise to add
    
    Returns:
        tuple: (data, conditions) where data is (N, 2) and conditions is (N, 1)
    """
    all_data = []
    all_conditions = []
    
    # Condition 0: Spiral distribution
    n_spiral = n_samples_per_class
    t = np.linspace(0, 4 * np.pi, n_spiral)
    r = t / (4 * np.pi) * 2  # Radius grows from 0 to 2
    spiral_x = r * np.cos(t) + 2
    spiral_y = r * np.sin(t) + 2
    spiral_data = np.column_stack([spiral_x, spiral_y])
    spiral_data += np.random.normal(0, noise, spiral_data.shape)
    all_data.append(spiral_data)
    all_conditions.append(np.full((n_spiral, 1), 0))
    
    # Condition 1: Ring distribution (donut)
    n_ring = n_samples_per_class
    # Inner ring
    inner_theta = np.random.uniform(0, 2 * np.pi, n_ring // 2)
    inner_r = np.random.normal(1.5, 0.2, n_ring // 2)
    inner_x = inner_r * np.cos(inner_theta) - 2
    inner_y = inner_r * np.sin(inner_theta) + 2
    # Outer ring
    outer_theta = np.random.uniform(0, 2 * np.pi, n_ring // 2)
    outer_r = np.random.normal(3.0, 0.2, n_ring // 2)
    outer_x = outer_r * np.cos(outer_theta) - 2
    outer_y = outer_r * np.sin(outer_theta) + 2
    ring_data = np.vstack([np.column_stack([inner_x, inner_y]), 
                          np.column_stack([outer_x, outer_y])])
    ring_data += np.random.normal(0, noise, ring_data.shape)
    all_data.append(ring_data)
    all_conditions.append(np.full((n_ring, 1), 1))
    
    # Condition 2: Multi-modal distribution (3 clusters)
    n_multimodal = n_samples_per_class
    # Cluster 1
    cluster1 = np.random.multivariate_normal([2, -2], [[0.3, 0.1], [0.1, 0.3]], n_multimodal // 3)
    # Cluster 2
    cluster2 = np.random.multivariate_normal([4, -1], [[0.2, -0.1], [-0.1, 0.2]], n_multimodal // 3)
    # Cluster 3
    cluster3 = np.random.multivariate_normal([1, -4], [[0.4, 0.0], [0.0, 0.4]], n_multimodal - 2 * (n_multimodal // 3))
    multimodal_data = np.vstack([cluster1, cluster2, cluster3])
    multimodal_data += np.random.normal(0, noise, multimodal_data.shape)
    all_data.append(multimodal_data)
    all_conditions.append(np.full((n_multimodal, 1), 2))
    
    # Condition 3: L-shaped distribution
    n_lshape = n_samples_per_class
    # Vertical part
    vertical_x = np.random.normal(-2, 0.2, n_lshape // 2)
    vertical_y = np.random.uniform(-4, 1, n_lshape // 2)
    # Horizontal part
    horizontal_x = np.random.uniform(-2, 1, n_lshape - n_lshape // 2)
    horizontal_y = np.random.normal(-2, 0.2, n_lshape - n_lshape // 2)
    lshape_data = np.vstack([np.column_stack([vertical_x, vertical_y]), 
                            np.column_stack([horizontal_x, horizontal_y])])
    lshape_data += np.random.normal(0, noise, lshape_data.shape)
    all_data.append(lshape_data)
    all_conditions.append(np.full((n_lshape, 1), 3))
    
    # Combine all data
    data = np.vstack(all_data)
    conditions = np.vstack(all_conditions)
    
    # Shuffle
    indices = np.random.permutation(len(data))
    data = data[indices]
    conditions = conditions[indices]
    
    return torch.tensor(data, dtype=torch.float32), torch.tensor(conditions, dtype=torch.float32)


def plot_conditional_data(data, conditions, save_path=None):
    """
    Plot conditional data with different colors for each condition
    
    Args:
        data: Data tensor (N, 2)
        conditions: Condition tensor (N, 1)
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    data_np = data.detach().cpu().numpy()
    conditions_np = conditions.detach().cpu().numpy().flatten()
    
    # Plot all data together
    axes[0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, s=1)
    axes[0].set_title('All Complex Conditional Data')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].grid(True, alpha=0.3)
    
    # Plot data colored by condition
    colors = ['red', 'blue', 'green', 'orange']
    condition_names = ['Spiral', 'Ring', 'Multi-modal', 'L-shape']
    
    for i in range(4):
        mask = conditions_np == i
        if np.any(mask):
            axes[1].scatter(data_np[mask, 0], data_np[mask, 1], 
                          alpha=0.6, s=1, color=colors[i], label=condition_names[i])
    
    axes[1].set_title('Complex Distributions by Condition')
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_source_distributions(target_dim=2, n_samples=2000, save_path=None):
    """
    Plot different source distributions for comparison
    
    Args:
        target_dim: Dimension of the data
        n_samples: Number of samples to generate
        save_path: Optional path to save the plot
    """
    # Create source samplers
    source_samplers = {
        'gaussian': gaussian_sampler(target_dim),
        'uniform': uniform_sampler(target_dim),
        'mixture': mixture_sampler(target_dim),
        'ring': ring_sampler(target_dim)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (source_name, sampler) in enumerate(source_samplers.items()):
        # Generate samples
        with torch.no_grad():
            samples = sampler(n_samples, 'cpu').numpy()
        
        # Plot
        axes[i].scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=1)
        axes[i].set_title(f'Source: {source_name.title()}')
        axes[i].set_xlabel('X1')
        axes[i].set_ylabel('X2')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(-4, 4)
        axes[i].set_ylim(-4, 4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_spiral_comparison(source_data, target_data, generated_data, save_path=None):
    """
    Plot comparison of spiral source, target, and generated data
    
    Args:
        source_data: Source spiral data tensor (N, 2)
        target_data: Target spiral data tensor (N, 2)
        generated_data: Generated data tensor (N, 2)
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert to numpy for plotting
    source_np = source_data.detach().cpu().numpy()
    target_np = target_data.detach().cpu().numpy()
    generated_np = generated_data.detach().cpu().numpy()
    
    # Source data
    axes[0].scatter(source_np[:, 0], source_np[:, 1], alpha=0.6, s=1, color='blue')
    axes[0].set_title('Source Spiral')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Target data
    axes[1].scatter(target_np[:, 0], target_np[:, 1], alpha=0.6, s=1, color='red')
    axes[1].set_title('Target Spiral')
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    # Generated data
    axes[2].scatter(generated_np[:, 0], generated_np[:, 1], alpha=0.6, s=1, color='green')
    axes[2].set_title('Generated Spiral')
    axes[2].set_xlabel('X1')
    axes[2].set_ylabel('X2')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_conditional_generation(flow_model, test_data, test_conditions, device='cpu', save_path=None):
    """
    Plot conditional generation results
    
    Args:
        flow_model: Trained conditional ConditionalFlowModel
        test_data: Test data tensor (N, 2)
        test_conditions: Test conditions tensor (N, 1)
        device: Device to run on
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    test_data_np = test_data.detach().cpu().numpy()
    test_conditions_np = test_conditions.detach().cpu().numpy().flatten()
    
    colors = ['red', 'blue', 'green', 'orange']
    condition_names = ['Spiral', 'Ring', 'Multi-modal', 'L-shape']
    
    # Generate samples for each condition
    with torch.no_grad():
        for condition in range(4):
            # Get test data for this condition
            mask = test_conditions_np == condition
            if not np.any(mask):
                continue
                
            condition_data = test_data[mask]
            condition_tensor = torch.full((len(condition_data), 1), float(condition), device=device, dtype=torch.float32)
            
            # Generate samples
            generated = flow_model.predict(
                batch_size=len(condition_data),
                condition=condition_tensor,
                device=device,
                num_steps=100
            )
            
            generated_np = generated.detach().cpu().numpy()
            
            # Plot target data
            axes[0, condition].scatter(condition_data[:, 0], condition_data[:, 1], 
                                    alpha=0.6, s=1, color=colors[condition])
            axes[0, condition].set_title(f'Target - {condition_names[condition]}')
            axes[0, condition].set_xlabel('X1')
            axes[0, condition].set_ylabel('X2')
            axes[0, condition].grid(True, alpha=0.3)
            
            # Plot generated data
            axes[1, condition].scatter(generated_np[:, 0], generated_np[:, 1], 
                                     alpha=0.6, s=1, color=colors[condition])
            axes[1, condition].set_title(f'Generated - {condition_names[condition]}')
            axes[1, condition].set_xlabel('X1')
            axes[1, condition].set_ylabel('X2')
            axes[1, condition].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def run_experiment(model_type='fm', epochs=100, batch_size=256, learning_rate=1e-4, n_samples=5000):
    """
    Create a flow model

    Create target data which is a bimodal 2d distribution

    Train the flow model to predict the target data

    Plot the target data and the predicted data
    """
    print("Starting Flow Matching Experiment...")
    
    # Create plots directory
    plots_dir = "test_plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Created plots directory: {plots_dir}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate bimodal 2D data
    print("Generating bimodal 2D data...")
    target_data = generate_bimodal_data(n_samples=n_samples, noise=0.1)
    print(f"Generated {target_data.shape[0]} samples")
    
    # Split data into train/validation/test using torch
    n_samples = target_data.shape[0]
    indices = torch.randperm(n_samples)
    
    # 60% train, 20% validation, 20% test
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = target_data[train_indices]
    val_data = target_data[val_indices]
    test_data = target_data[test_indices]
    
    print(f"Train samples: {train_data.shape[0]}")
    print(f"Validation samples: {val_data.shape[0]}")
    print(f"Test samples: {test_data.shape[0]}")
    
    # Create flow matching model
    print("Creating flow matching model...")
    target_dim = 2
    if model_type == 'fm':
        flow_model = ConditionalFlowModel(
            target_dim=target_dim,
            cond_dim=0,  # No additional conditioning
            source_sampler=gaussian_sampler(target_dim),  # Default Gaussian source
            model_type='unet',
            use_spectral_norm=True
        )
    elif model_type == 'eqm':

        flow_model = EquilibriumMatchingModel(
            target_dim=target_dim,
            cond_dim=0,  # No additional conditioning
            source_sampler=gaussian_sampler(target_dim),  # Default Gaussian source
            lambda_=1.0,
            gamma_type='linear',
            objective_type='implicit',
            energy_type='dot',
            grad_type='nag',
            eta=0.02,
            model_type='transformer'
        )
    
    print(f"Model parameters: {sum(p.numel() for p in flow_model.parameters()):,}")

    
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Train the model
    print("Training the model...")
    history = train_flow_model(
        flow_model=flow_model,
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        verbose=True
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history, save_path=os.path.join(plots_dir, 'training_history.png'))
    
    # Evaluate the model
    print("Evaluating the model...")
    metrics = evaluate_flow_model(flow_model, test_data, num_samples=1000, device=device)
    
    print("Evaluation metrics:")
    print(f"  Mean MSE: {metrics['mean_mse']:.6f}")
    print(f"  Std MSE: {metrics['std_mse']:.6f}")
    print(f"  Mean Wasserstein Distance: {metrics['mean_wasserstein']:.6f}")
    print(f"  Max Wasserstein Distance: {metrics['max_wasserstein']:.6f}")
    
    # Generate samples for visualization
    print("Generating samples for visualization...")
    with torch.no_grad():
        generated_samples = flow_model.predict(
            batch_size=1000,
            device=device,
            num_steps=100
        )
    
    # Plot comparison
    print("Creating comparison plots...")
    plot_data_comparison(
        target_data=test_data,
        generated_data=generated_samples,
        save_path=os.path.join(plots_dir, 'data_comparison.png')
    )
    
    print("Experiment completed successfully!")
    print(f"All plots saved in '{plots_dir}' directory:")
    print(f"  - training_history.png")
    print(f"  - data_comparison.png")


def run_conditional_experiment():
    """
    Create a conditional flow model with categorical conditioning
    
    Create target data with 4 different modes based on condition
    
    Train the conditional flow model to predict the target data
    
    Plot the conditional generation results
    """
    print("Starting Conditional Flow Matching Experiment...")
    
    # Create plots directory
    plots_dir = "test_plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Created plots directory: {plots_dir}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate conditional 2D data
    print("Generating conditional 2D data...")
    target_data, conditions = generate_conditional_data(n_samples_per_class=10000, noise=0.1)
    print(f"Generated {target_data.shape[0]} samples with {len(torch.unique(conditions))} conditions")
    
    # Split data into train/validation/test using torch
    n_samples = target_data.shape[0]
    indices = torch.randperm(n_samples)
    
    # 60% train, 20% validation, 20% test
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = target_data[train_indices]
    train_conditions = conditions[train_indices]
    val_data = target_data[val_indices]
    val_conditions = conditions[val_indices]
    test_data = target_data[test_indices]
    test_conditions = conditions[test_indices]
    
    print(f"Train samples: {train_data.shape[0]}")
    print(f"Validation samples: {val_data.shape[0]}")
    print(f"Test samples: {test_data.shape[0]}")
    
    # Plot conditional data
    print("Plotting conditional data...")
    plot_conditional_data(
        data=target_data,
        conditions=conditions,
        save_path=os.path.join(plots_dir, 'conditional_data.png')
    )
    
    # Create conditional flow matching model
    print("Creating conditional flow matching model...")
    target_dim = 2
    cond_dim = 1  # Categorical condition
    flow_model = ConditionalFlowModel(
        target_dim=target_dim,
        cond_dim=cond_dim,
        source_sampler=gaussian_sampler(target_dim),  # Use mixture source for more interesting flow
        model_type='unet',
        use_spectral_norm=True
    )

    # flow_model = LatentBridgeModel(
    #     target_dim=target_dim,
    #     bridge_noise_sigma=0.01,
    #     cond_dim=cond_dim,
    #     hidden_channels=128,
    #     num_layers=4,
    #     source_sampler=mixture_sampler(target_dim)  # Use mixture source for more interesting flow
    # )
    
    print(f"Model parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    
    # Training configuration
    epochs = 100
    batch_size = 256
    learning_rate = 1e-3
    
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Train the model
    print("Training the conditional model...")
    history = train_flow_model(
        flow_model=flow_model,
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        verbose=True,
        train_conditions=train_conditions,
        val_conditions=val_conditions
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history, save_path=os.path.join(plots_dir, 'conditional_training_history.png'))
    
    # Plot conditional generation results
    print("Creating conditional generation plots...")
    plot_conditional_generation(
        flow_model=flow_model,
        test_data=test_data,
        test_conditions=test_conditions,
        device=device,
        save_path=os.path.join(plots_dir, 'conditional_generation.png')
    )
    
    print("Conditional experiment completed successfully!")
    print(f"All plots saved in '{plots_dir}' directory:")
    print(f"  - conditional_data.png")
    print(f"  - conditional_training_history.png")
    print(f"  - conditional_generation.png")


def run_spiral_experiment():
    """
    Spiral-to-spiral flow matching experiment
    
    Learn to transform from one spiral distribution to another with different parameters
    """
    print("Starting Spiral-to-Spiral Flow Matching Experiment...")
    
    # Create plots directory
    plots_dir = "test_plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Created plots directory: {plots_dir}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Define spiral parameters
    source_params = {
        'center': (0, 0),
        'turns': 2.0,
        'radius_max': 2.0,
        'noise': 0.1
    }
    
    target_params = {
        'center': (1, 1),  # Shifted center
        'turns': 2.5,      # More turns
        'radius_max': 2.5, # Larger radius
        'noise': 0.1     # More noise
    }
    
    print(f"Source spiral: center={source_params['center']}, turns={source_params['turns']}, radius_max={source_params['radius_max']}")
    print(f"Target spiral: center={target_params['center']}, turns={target_params['turns']}, radius_max={target_params['radius_max']}")
    
    # Generate target data
    print("Generating target spiral data...")
    target_data = generate_spiral_target_data(
        n_samples=5000,
        center=target_params['center'],
        turns=target_params['turns'],
        radius_max=target_params['radius_max'],
        noise=target_params['noise']
    )
    
    # Split data into train/validation/test
    n_samples = target_data.shape[0]
    indices = torch.randperm(n_samples)
    
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = target_data[train_indices]
    val_data = target_data[val_indices]
    test_data = target_data[test_indices]
    
    print(f"Train samples: {train_data.shape[0]}")
    print(f"Validation samples: {val_data.shape[0]}")
    print(f"Test samples: {test_data.shape[0]}")
    
    # # Create source sampler
    # source_sampler = spiral_sampler(
    #     target_dim=2,
    #     center=source_params['center'],
    #     turns=source_params['turns'],
    #     radius_max=source_params['radius_max'],
    #     noise=source_params['noise']
    # )

    source_sampler = gaussian_sampler(target_dim=2)
    
    # Generate source data for visualization
    print("Generating source spiral data for visualization...")
    with torch.no_grad():
        source_data = source_sampler(1000, device)
    
    # Create flow matching model
    print("Creating spiral flow matching model...")
    flow_model = ConditionalFlowModel(
        target_dim=2,
        cond_dim=0,
        source_sampler=source_sampler
    )
    
    print(f"Model parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    
    # Training configuration
    epochs = 500  # More epochs for complex transformation
    batch_size = 256
    learning_rate = 1e-3
    
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Train the model
    print("Training the spiral flow model...")
    history = train_flow_model(
        flow_model=flow_model,
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        verbose=True
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history, save_path=os.path.join(plots_dir, 'spiral_training_history.png'))
    
    # Generate samples for visualization
    print("Generating samples for visualization...")
    with torch.no_grad():
        generated_samples = flow_model.predict(
            batch_size=1000,
            device=device,
            prior_sampler=source_sampler,
        )
    
    # Plot spiral comparison
    print("Creating spiral comparison plots...")
    plot_spiral_comparison(
        source_data=source_data,
        target_data=test_data,
        generated_data=generated_samples,
        save_path=os.path.join(plots_dir, 'spiral_comparison.png')
    )
    
    # Evaluate the model
    print("Evaluating the spiral model...")
    metrics = evaluate_flow_model(flow_model, test_data, num_samples=1000, device=device)
    
    print("Spiral experiment evaluation metrics:")
    print(f"  Mean MSE: {metrics['mean_mse']:.6f}")
    print(f"  Std MSE: {metrics['std_mse']:.6f}")
    print(f"  Mean Wasserstein Distance: {metrics['mean_wasserstein']:.6f}")
    print(f"  Max Wasserstein Distance: {metrics['max_wasserstein']:.6f}")
    
    print("Spiral experiment completed successfully!")
    print(f"All plots saved in '{plots_dir}' directory:")
    print(f"  - spiral_training_history.png")
    print(f"  - spiral_comparison.png")


def run_source_comparison_experiment():
    """
    Compare different source distributions on the same target data
    """
    print("Starting Source Distribution Comparison Experiment...")
    
    # Create plots directory
    plots_dir = "test_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate simple target data (bimodal)
    print("Generating target data...")
    target_data = generate_bimodal_data(n_samples=2000, noise=0.1)
    
    # Split data
    n_samples = target_data.shape[0]
    indices = torch.randperm(n_samples)
    train_size = int(0.8 * n_samples)
    train_data = target_data[indices[:train_size]]
    test_data = target_data[indices[train_size:]]
    
    # Test different source distributions
    source_samplers = {
        'gaussian': gaussian_sampler(2),
        'uniform': uniform_sampler(2),
        'mixture': mixture_sampler(2),
        'ring': ring_sampler(2)
    }
    results = {}
    
    for source_name, source_sampler in source_samplers.items():
        print(f"\nTraining with {source_name} source distribution...")
        
        # Create model
        flow_model = ConditionalFlowModel(
            target_dim=2,
            cond_dim=0,
            source_sampler=source_sampler
        )
        
        # Train model
        history = train_flow_model(
            flow_model=flow_model,
            train_data=train_data,
            val_data=test_data,
            epochs=50,  # Shorter training for comparison
            batch_size=128,
            learning_rate=1e-3,
            device=device,
            verbose=False
        )
        
        # Generate samples
        with torch.no_grad():
            generated_samples = flow_model.predict(
                batch_size=500,
                device=device,
                num_steps=100
            )
        
        results[source_name] = {
            'model': flow_model,
            'generated': generated_samples,
            'final_loss': history['val_loss'][-1]
        }
    
    # Plot comparison
    print("Creating source comparison plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot target data
    target_np = test_data.detach().cpu().numpy()
    axes[0, 0].scatter(target_np[:, 0], target_np[:, 1], alpha=0.6, s=1)
    axes[0, 0].set_title('Target Data (Bimodal)')
    axes[0, 0].set_xlabel('X1')
    axes[0, 0].set_ylabel('X2')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot generated data for each source
    for i, (source_name, result) in enumerate(results.items()):
        row = 0 if i < 2 else 1
        col = (i % 2) + 1
        
        generated_np = result['generated'].detach().cpu().numpy()
        axes[row, col].scatter(generated_np[:, 0], generated_np[:, 1], alpha=0.6, s=1)
        axes[row, col].set_title(f'Generated ({source_name.title()})\nLoss: {result["final_loss"]:.4f}')
        axes[row, col].set_xlabel('X1')
        axes[row, col].set_ylabel('X2')
        axes[row, col].grid(True, alpha=0.3)
    
    # Remove unused subplot
    fig.delaxes(axes[1, 0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'source_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Source comparison experiment completed!")
    print(f"Results saved in '{plots_dir}/source_comparison.png'")



def _sample_gaussian(center, cov_scale, n_samples):
    mean = np.array(center)
    cov = np.array([[cov_scale, 0.0], [0.0, cov_scale]], dtype=float)
    return np.random.multivariate_normal(mean, cov, n_samples)


def _make_quadrant_datasets(n_per_mode=5000, cov_scale=0.05):
    """
    Build source/target quadrant Gaussian datasets and paired/unpaired prior conditions.

    Source modes: [-1,  1] (Q2), [-1, -1] (Q3)
    Target modes: [ 1,  1] (Q1), [ 1, -1] (Q4)

    Paired mapping: Q3 -> Q1, Q2 -> Q4 (diagonals)
    Unpaired mapping for Q4: prior can be Q2 or Q3 (random)
    """
    # Target clusters
    target_q1 = _sample_gaussian([1.0, 1.0], cov_scale, n_per_mode)
    target_q4 = _sample_gaussian([1.0, -1.0], cov_scale, n_per_mode)
    target_data = np.vstack([target_q1, target_q4])
    target_labels = np.concatenate([np.zeros(n_per_mode, dtype=int), np.ones(n_per_mode, dtype=int)])  # 0 -> Q1, 1 -> Q4

    # Priors (as conditions) for paired (use continuous coordinates around centers)
    prior_q3 = _sample_gaussian([-1.0, -1.0], cov_scale, n_per_mode)  # for Q1
    prior_q2 = _sample_gaussian([-1.0, 1.0], cov_scale, n_per_mode)   # for Q4
    paired_prior = np.vstack([prior_q3, prior_q2])  # aligned with target_data order

    # Priors (as conditions) for unpaired
    # For Q1 keep Q3 as before to avoid ambiguity; for Q4 randomly choose Q2 or Q3
    rng = np.random.default_rng()
    choice = rng.integers(0, 2, size=n_per_mode)  # 0 or 1
    mixed_q4 = np.where(choice[:, None] == 0, _sample_gaussian([-1.0, 1.0], cov_scale, n_per_mode), _sample_gaussian([-1.0, -1.0], cov_scale, n_per_mode))
    unpaired_prior = np.vstack([prior_q3, mixed_q4])

    return (
        torch.tensor(target_data, dtype=torch.float32),
        torch.tensor(paired_prior, dtype=torch.float32),
        torch.tensor(unpaired_prior, dtype=torch.float32),
        torch.tensor(target_labels, dtype=torch.long),
    )


def _q23_source_sampler(std=0.05):
    """Return a sampler that draws from a mixture of the two prior modes (Q2 and Q3) independent of condition."""
    means = np.array([[-1.0, 1.0],  # Q2
                      [-1.0, -1.0]])  # Q3
    cov = np.array([[std, 0.0], [0.0, std]])

    def sampler(batch_size, device):
        # Randomly choose components
        comps = np.random.randint(0, 2, size=batch_size)
        samples = np.zeros((batch_size, 2), dtype=np.float32)
        for i in range(batch_size):
            samples[i] = np.random.multivariate_normal(means[comps[i]], cov)
        return torch.tensor(samples, dtype=torch.float32, device=device)

    return sampler


def _train_conditional(flow_model, train_data, val_data, train_cond, val_cond, device, epochs=200, batch_size=256, lr=1e-3, verbose=True):
    history = train_flow_model(
        flow_model=flow_model,
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        device=device,
        verbose=verbose,
        train_conditions=train_cond,
        val_conditions=val_cond,
    )
    return history


def _plot_steering_results(target_data, paired_gen, unpaired_gen, title_suffix, save_path=None, prior_samples=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    tgt = target_data.detach().cpu().numpy()
    paired_np = paired_gen.detach().cpu().numpy()
    unpaired_np = unpaired_gen.detach().cpu().numpy()
    prior_np = None if prior_samples is None else prior_samples.detach().cpu().numpy()

    # Target with prior overlay (if provided)
    axes[0].scatter(tgt[:, 0], tgt[:, 1], s=4, alpha=0.5, label='Target', color='C0')
    if prior_np is not None:
        axes[0].scatter(prior_np[:, 0], prior_np[:, 1], s=6, alpha=0.7, label='Prior', color='C1')
        axes[0].legend()
    axes[0].set_title('Target' + (" + Prior" if prior_np is not None else ""))
    axes[0].grid(True, alpha=0.3)

    # Paired
    if prior_np is not None:
        axes[1].scatter(prior_np[:, 0], prior_np[:, 1], s=6, alpha=0.4, label='Prior', color='C1')
    axes[1].scatter(paired_np[:, 0], paired_np[:, 1], s=4, alpha=0.7, color='green', label='Generated')
    axes[1].set_title(f'Paired prior → Generated ({title_suffix})')
    axes[1].grid(True, alpha=0.3)
    if prior_np is not None:
        axes[1].legend()

    # Unpaired
    if prior_np is not None:
        axes[2].scatter(prior_np[:, 0], prior_np[:, 1], s=6, alpha=0.4, label='Prior', color='C1')
    axes[2].scatter(unpaired_np[:, 0], unpaired_np[:, 1], s=4, alpha=0.7, color='red', label='Generated')
    axes[2].set_title(f'Unpaired prior → Generated ({title_suffix})')
    axes[2].grid(True, alpha=0.3)
    if prior_np is not None:
        axes[2].legend()

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.axvline(0, color='k', lw=0.5, alpha=0.3)
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def run_prior_steering_experiment():
    """
    Does learning from a prior allow steering with samples from the prior?

    - Source (prior) modes at [-1, 1] (Q2) and [-1, -1] (Q3)
    - Target modes at [ 1, 1] (Q1) and [ 1, -1] (Q4)
    - Pair diagonal quadrants: Q3→Q1, Q2→Q4
    - Train two conditional flow models with cond_dim=2 (continuous prior coords):
        1) Paired prior (deterministic diagonal pairing)
        2) Unpaired prior (Q4 targets see mixed Q2/Q3 priors)
    - At test: sample prior from Q3; expect Paired→Q1, Unpaired→Q4 (shortest flow)
    """
    print("Starting Prior-Steering Experiment...")

    plots_dir = "test_plots"
    os.makedirs(plots_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Build datasets
    target_data, paired_prior, unpaired_prior, target_labels = _make_quadrant_datasets(n_per_mode=6000, cov_scale=0.05)

    # Split train/val/test (60/20/20) keeping alignment across tensors
    n = target_data.shape[0]
    indices = torch.randperm(n)
    train_n = int(0.6 * n)
    val_n = int(0.2 * n)

    train_idx = indices[:train_n]
    val_idx = indices[train_n:train_n + val_n]
    test_idx = indices[train_n + val_n:]

    train_tgt, val_tgt, test_tgt = target_data[train_idx], target_data[val_idx], target_data[test_idx]
    train_paired_c, val_paired_c, test_paired_c = paired_prior[train_idx], paired_prior[val_idx], paired_prior[test_idx]
    train_unpaired_c, val_unpaired_c, test_unpaired_c = unpaired_prior[train_idx], unpaired_prior[val_idx], unpaired_prior[test_idx]

    # Define two conditional models (cond_dim=2 for prior coordinates as condition),
    # but the source sampler is independent mixture over Q2/Q3
    src_sampler = _q23_source_sampler(std=0.1)

    def build_model():
        return ConditionalFlowModel(
            target_dim=2,
            cond_dim=2,
            source_sampler=src_sampler,
        )

    paired_model = build_model()
    unpaired_model = build_model()

    print(f"Paired model params: {sum(p.numel() for p in paired_model.parameters()):,}")
    print(f"Unpaired model params: {sum(p.numel() for p in unpaired_model.parameters()):,}")

    # Train
    print("Training paired model (diagonal pairing)...")
    # Train paired: pass prior coords as conditions to the network; source is sampled from mixture independent of condition
    hist_paired = _train_conditional(
        paired_model,
        train_tgt,
        val_tgt,
        train_paired_c,
        val_paired_c,
        device,
        epochs=200,
        batch_size=256,
        lr=1e-3,
        verbose=True,
    )

    # Train unpaired: pass mixed prior conditions; source still sampled from mixture independent of condition
    hist_unpaired = _train_conditional(
        unpaired_model,
        train_tgt,
        val_tgt,
        train_unpaired_c,
        val_unpaired_c,
        device,
        epochs=200,
        batch_size=256,
        lr=1e-3,
        verbose=True,
    )

    # Plot training curves
    plot_training_history(hist_paired, save_path=os.path.join(plots_dir, 'prior_paired_training.png'))
    plot_training_history(hist_unpaired, save_path=os.path.join(plots_dir, 'prior_unpaired_training.png'))

    # Steering test: prior from Q3 should map to Q1 for the paired model
    with torch.no_grad():
        test_batch = 4000
        # Condition indicating Q3; actual source will be sampled from mixture
        prior_q3_test = torch.tensor(_sample_gaussian([-1.0, -1.0], 0.05, test_batch), dtype=torch.float32, device=device)

        paired_gen = paired_model.predict(batch_size=test_batch, condition=prior_q3_test, device=device, num_steps=100)
        unpaired_gen = unpaired_model.predict(batch_size=test_batch, condition=prior_q3_test, device=device, num_steps=100)

    _plot_steering_results(
        target_data=test_tgt,
        paired_gen=paired_gen,
        unpaired_gen=unpaired_gen,
        title_suffix='prior from Q3',
        save_path=os.path.join(plots_dir, 'prior_steering_q3.png'),
        prior_samples=prior_q3_test,
    )

    # Additionally show steering with prior from Q2 (expect paired→Q4)
    with torch.no_grad():
        # Condition indicating Q2; actual source will be sampled from mixture
        prior_q2_test = torch.tensor(_sample_gaussian([-1.0, 1.0], 0.05, 4000), dtype=torch.float32, device=device)
        paired_gen_q2 = paired_model.predict(batch_size=4000, condition=prior_q2_test, device=device, num_steps=100)
        unpaired_gen_q2 = unpaired_model.predict(batch_size=4000, condition=prior_q2_test, device=device, num_steps=100)

    _plot_steering_results(
        target_data=test_tgt,
        paired_gen=paired_gen_q2,
        unpaired_gen=unpaired_gen_q2,
        title_suffix='prior from Q2',
        save_path=os.path.join(plots_dir, 'prior_steering_q2.png'),
        prior_samples=prior_q2_test,
    )

    print("Prior-steering experiment completed. Plots saved to 'plots/'.")


def run_prior_vs_paired_simple_experiment():
    """
    Simple experiment:
    - Compare training with a sampleable prior (source sampler over Q2/Q3 mixture)
      vs a paired prior (precomputed prior points paired with targets and used as sources).
    - No network conditioning; only the source distribution differs.
    """
    print("Starting Simple Prior vs Paired-Prior Experiment...")

    plots_dir = "test_plots"
    os.makedirs(plots_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Data: Q1 and Q4 targets; paired prior built by randomly sampling from {Q2,Q3} for each target
    n_per_mode = 4000
    cov_scale = 0.05
    target_q1 = _sample_gaussian([1.0, 1.0], cov_scale, n_per_mode)
    target_q4 = _sample_gaussian([1.0, -1.0], cov_scale, n_per_mode)
    target_np = np.vstack([target_q1, target_q4])

    # Randomly choose prior component (Q2 or Q3) for every target, sample and pair up front
    rng = np.random.default_rng()
    comps = rng.integers(0, 2, size=target_np.shape[0])  # 0->Q2, 1->Q3
    prior_means = np.where(comps[:, None] == 0, np.array([-1.0, 1.0]), np.array([-1.0, -1.0]))
    paired_prior_np = np.array([
        np.random.multivariate_normal(prior_means[i], [[cov_scale, 0.0], [0.0, cov_scale]])
        for i in range(target_np.shape[0])
    ], dtype=np.float32)

    target_data = torch.tensor(target_np, dtype=torch.float32)
    paired_prior = torch.tensor(paired_prior_np, dtype=torch.float32)

    # Train/val split
    n = target_data.shape[0]
    indices = torch.randperm(n)
    train_n = int(0.8 * n)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]

    train_tgt, val_tgt = target_data[train_idx], target_data[val_idx]
    train_paired_c, val_paired_c = paired_prior[train_idx], paired_prior[val_idx]

    # Model A: Sampleable prior (mixture source)
    sampler_src = _q23_source_sampler(std=0.1)
    model_sampler = ConditionalFlowModel(
        target_dim=2,
        cond_dim=0,
        source_sampler=sampler_src,
    )
    model_sampler.use_condition_for_source = False  # ignore any condition for source

    # Model B: Paired prior used as source (no conditioning in network)
    model_paired = ConditionalFlowModel(
        target_dim=2,
        cond_dim=0,
        source_sampler=gaussian_sampler(2),  # fallback; actual source comes from condition
    )

    # Train both
    print("Training model with sampleable prior (mixture source)...")
    hist_sampler = train_flow_model(
        flow_model=model_sampler,
        train_data=train_tgt,
        val_data=val_tgt,
        epochs=500,
        batch_size=256,
        learning_rate=1e-3,
        device=device,
        verbose=True,
        train_prior_sampler=sampler_src,
        val_prior_sampler=sampler_src,
    )

    print("Training model with paired prior (precomputed sources)...")
    hist_paired = train_flow_model(
        flow_model=model_paired,
        train_data=train_tgt,
        val_data=val_tgt,
        epochs=500,
        batch_size=256,
        learning_rate=1e-3,
        device=device,
        verbose=True,
        train_prior_samples=train_paired_c,  # explicitly use paired sources
        val_prior_samples=val_paired_c,
    )

    # Plot training histories
    plot_training_history(hist_sampler, save_path=os.path.join(plots_dir, 'simple_prior_sampler_training.png'))
    plot_training_history(hist_paired, save_path=os.path.join(plots_dir, 'simple_paired_prior_training.png'))

    # Generate samples for visualization and collect priors used
    with torch.no_grad():
        # Validation-sized priors for fair comparison
        prior_sampler_val = sampler_src(val_tgt.shape[0], device)
        gen_sampler = model_sampler.predict(batch_size=val_tgt.shape[0], device=device, num_steps=100, prior_sampler=sampler_src)

        prior_paired_val = val_paired_c.to(device)
        gen_paired = model_paired.predict(batch_size=val_tgt.shape[0], device=device, num_steps=100, prior_samples=prior_paired_val)

    # Side-by-side plots against targets with prior overlay
    def _plot_side_by_side_with_prior(title, generated, prior, save_name):
        plt.figure(figsize=(12, 4))
        tgt_np = val_tgt.detach().cpu().numpy()
        gen_np = generated.detach().cpu().numpy()
        prior_np = prior.detach().cpu().numpy()
        # Target + Prior
        plt.subplot(1, 3, 1)
        plt.scatter(tgt_np[:, 0], tgt_np[:, 1], s=3, alpha=0.6, label='Target', color='C0')
        plt.scatter(prior_np[:, 0], prior_np[:, 1], s=4, alpha=0.6, label='Prior', color='C1')
        plt.title('Target (val) + Prior')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Generated + Prior
        plt.subplot(1, 3, 2)
        plt.scatter(gen_np[:, 0], gen_np[:, 1], s=3, alpha=0.7, label='Generated', color='green')
        plt.scatter(prior_np[:, 0], prior_np[:, 1], s=4, alpha=0.4, label='Prior', color='C1')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Prior only
        plt.subplot(1, 3, 3)
        plt.scatter(prior_np[:, 0], prior_np[:, 1], s=3, alpha=0.7, color='C1')
        plt.title('Prior only')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()

    _plot_side_by_side_with_prior('Generated (sampleable prior)', gen_sampler, prior_sampler_val, 'simple_prior_sampler_generated.png')
    _plot_side_by_side_with_prior('Generated (paired prior)', gen_paired, prior_paired_val, 'simple_paired_prior_generated.png')

    print("Simple prior vs paired-prior experiment complete. Plots saved to 'plots/'.")



def run_equilibrium_hyperparameter_sweep():
    """
    Run hyperparameter sweep for EquilibriumMatchingModel
    
    Sweep over:
    - learning_rate: [1e-4, 5e-4, 1e-3, 5e-3]
    - gamma_type: ['linear', 'truncated']
    - lambda_: [0.5, 1.0, 2.0]
    """
    print("Starting EquilibriumMatchingModel Hyperparameter Sweep...")
    
    # Create plots directory
    plots_dir = "test_plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Created plots directory: {plots_dir}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate bimodal 2D data
    print("Generating bimodal 2D data...")
    target_data = generate_bimodal_data(n_samples=5000, noise=0.1)
    print(f"Generated {target_data.shape[0]} samples")
    
    # Split data into train/validation/test
    n_samples = target_data.shape[0]
    indices = torch.randperm(n_samples)
    
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = target_data[train_indices]
    val_data = target_data[val_indices]
    test_data = target_data[test_indices]
    
    print(f"Train samples: {train_data.shape[0]}")
    print(f"Validation samples: {val_data.shape[0]}")
    print(f"Test samples: {test_data.shape[0]}")
    
    # # Hyperparameter grid
    # learning_rates = [1e-4]
    # gamma_types = ['linear', 'truncated']
    # lambda_values = [1.0, 4.0]
    # objective_types = ['implicit', 'explicit']
    # energy_types = ['dot', 'l2']
    
    # # Inference parameter grid (eta and mu for NAG sampling)
    # eta_values = [0.003, 0.03]
    # mu_values = [.035, .35]
    # grad_types = ['gd', 'nag']

    # Hyperparameter grid
    learning_rates = [1e-4]
    gamma_types = ['linear']
    lambda_values = [4.0]
    objective_types = ['implicit']
    energy_types = ['dot']
    
    # Inference parameter grid (eta and mu for NAG sampling)
    eta_values = np.linspace(0.003, 0.3, 50)
    mu_values = [.03, .35]
    grad_types = ['nag', 'gd']
    
    
    # Results storage
    results = []
    
    # Training configuration
    epochs = 1000
    batch_size = 256
    target_dim = 2
    
    print(f"Hyperparameter sweep configuration:")
    print(f"  Learning rates: {learning_rates}")
    print(f"  Gamma types: {gamma_types}")
    print(f"  Lambda values: {lambda_values}")
    print(f"  Objective types: {objective_types}")
    print(f"  Energy types: {energy_types}")
    print(f"  Eta values (inference): {eta_values}")
    print(f"  Mu values (inference): {mu_values}")
    print(f"  Grad types (inference): {grad_types}")
    print(f"  Epochs per config: {epochs}")
    print(f"  Total training configs: {len(learning_rates) * len(gamma_types) * len(lambda_values) * len(objective_types) * len(energy_types)}")
    print(f"  Total inference configs per model: {len(eta_values) * len(mu_values) * len(grad_types)}")
    
    config_count = 0
    total_configs = len(learning_rates) * len(gamma_types) * len(lambda_values) * len(objective_types) * len(energy_types)
    
    for lr in learning_rates:
        for gamma_type in gamma_types:
            for lambda_ in lambda_values:
                for objective_type in objective_types:
                    for energy_type in energy_types:
                        config_count += 1
                        print(f"\n{'='*60}")
                        print(f"Configuration {config_count}/{total_configs}")
                        print(f"Learning Rate: {lr}")
                        print(f"Gamma Type: {gamma_type}")
                        print(f"Lambda: {lambda_}")
                        print(f"Objective Type: {objective_type}")
                        print(f"Energy Type: {energy_type}")
                        print(f"{'='*60}")
                        
                        # Create model with current hyperparameters
                        flow_model = EquilibriumMatchingModel(
                            target_dim=target_dim,
                            cond_dim=0,
                            source_sampler=gaussian_sampler(target_dim),
                            lambda_=lambda_,
                            gamma_type=gamma_type,
                            objective_type=objective_type,
                            energy_type=energy_type
                        )
                        
                        # Train the model
                        print("Training the model...")
                        history = train_flow_model(
                            flow_model=flow_model,
                            train_data=train_data,
                            val_data=val_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=lr,
                            device=device,
                            verbose=False  # Reduce output for sweep
                        )
                        
                        # Test different inference parameters
                        print("Testing inference parameters...")
                        inference_results = []
                        
                        for eta in eta_values:
                            for mu in mu_values:
                                for grad_type in grad_types:
                                    print(f"  Testing eta={eta}, mu={mu}, grad_type={grad_type}...")
                                    
                                    # Evaluate with specific eta, mu, and grad_type
                                    metrics = evaluate_flow_model(
                                        flow_model, 
                                        test_data, 
                                        num_samples=1000, 
                                        device=device,
                                        eta=eta,
                                        mu=mu,
                                        grad_type=grad_type
                                    )
                                    
                                    inference_result = {
                                        'eta': eta,
                                        'mu': mu,
                                        'grad_type': grad_type,
                                        'mean_mse': metrics['mean_mse'],
                                        'std_mse': metrics['std_mse'],
                                        'mean_wasserstein': metrics['mean_wasserstein'],
                                        'max_wasserstein': metrics['max_wasserstein']
                                    }
                                    inference_results.append(inference_result)
                        
                        # Find best inference parameters (by mean MSE)
                        best_inference = min(inference_results, key=lambda x: x['mean_mse'])
                        
                        # Store results
                        result = {
                            'learning_rate': lr,
                            'gamma_type': gamma_type,
                            'lambda_': lambda_,
                            'objective_type': objective_type,
                            'energy_type': energy_type,
                            'final_train_loss': history['train_loss'][-1],
                            'final_val_loss': history['val_loss'][-1],
                            'best_eta': best_inference['eta'],
                            'best_mu': best_inference['mu'],
                            'best_grad_type': best_inference['grad_type'],
                            'best_mean_mse': best_inference['mean_mse'],
                            'best_std_mse': best_inference['std_mse'],
                            'best_mean_wasserstein': best_inference['mean_wasserstein'],
                            'best_max_wasserstein': best_inference['max_wasserstein'],
                            'all_inference_results': inference_results
                        }
                        results.append(result)
                        
                        print(f"Results:")
                        print(f"  Final Train Loss: {result['final_train_loss']:.6f}")
                        print(f"  Final Val Loss: {result['final_val_loss']:.6f}")
                        print(f"  Best Inference (eta={best_inference['eta']}, mu={best_inference['mu']}, grad_type={best_inference['grad_type']}):")
                        print(f"    Mean MSE: {best_inference['mean_mse']:.6f}")
                        print(f"    Mean Wasserstein: {best_inference['mean_wasserstein']:.6f}")
    
    # Find best configuration
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SWEEP RESULTS")
    print(f"{'='*60}")
    
    # Sort by best inference MSE (lower is better)
    results_sorted = sorted(results, key=lambda x: x['best_mean_mse'])
    
    print("Top 5 configurations by best inference MSE:")
    for i, result in enumerate(results_sorted[:5]):
        print(f"{i+1}. LR={result['learning_rate']:.0e}, "
              f"Gamma={result['gamma_type']}, "
              f"Lambda={result['lambda_']:.1f}, "
              f"Obj={result['objective_type']}, "
              f"Energy={result['energy_type']}, "
              f"Eta={result['best_eta']:.2f}, "
              f"Mu={result['best_mu']:.2f}, "
              f"Grad={result['best_grad_type']} "
              f"-> MSE: {result['best_mean_mse']:.6f}")
    
    # Save results to file
    results_file = os.path.join(plots_dir, 'equilibrium_hyperparameter_sweep_results.txt')
    with open(results_file, 'w') as f:
        f.write("EquilibriumMatchingModel Hyperparameter Sweep Results\n")
        f.write("="*60 + "\n\n")
        
        f.write("All configurations (sorted by best inference MSE):\n")
        f.write("-" * 60 + "\n")
        for i, result in enumerate(results_sorted):
            f.write(f"{i+1:2d}. LR={result['learning_rate']:.0e}, "
                   f"Gamma={result['gamma_type']:8s}, "
                   f"Lambda={result['lambda_']:3.1f}, "
                   f"Obj={result['objective_type']:8s}, "
                   f"Energy={result['energy_type']:3s}, "
                   f"Eta={result['best_eta']:4.2f}, "
                   f"Mu={result['best_mu']:4.2f}, "
                   f"Grad={result['best_grad_type']:3s} | "
                   f"Val Loss: {result['final_val_loss']:.6f}, "
                   f"Best MSE: {result['best_mean_mse']:.6f}, "
                   f"Best Wasserstein: {result['best_mean_wasserstein']:.6f}\n")
        
        f.write("\nDetailed inference results for each configuration:\n")
        f.write("=" * 80 + "\n")
        for i, result in enumerate(results_sorted):
            f.write(f"\nConfig {i+1}: LR={result['learning_rate']:.0e}, "
                   f"Gamma={result['gamma_type']}, Lambda={result['lambda_']:.1f}, "
                   f"Obj={result['objective_type']}, Energy={result['energy_type']}\n")
            f.write("-" * 50 + "\n")
            for inf_result in result['all_inference_results']:
                f.write(f"  Eta={inf_result['eta']:4.2f}, Mu={inf_result['mu']:4.2f}, "
                       f"Grad={inf_result['grad_type']:3s} | "
                       f"MSE: {inf_result['mean_mse']:.6f}, "
                       f"Wasserstein: {inf_result['mean_wasserstein']:.6f}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Create visualization of best configuration
    best_config = results_sorted[0]
    print(f"\nVisualizing best configuration:")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Gamma Type: {best_config['gamma_type']}")
    print(f"  Lambda: {best_config['lambda_']}")
    print(f"  Objective Type: {best_config['objective_type']}")
    print(f"  Energy Type: {best_config['energy_type']}")
    print(f"  Best Eta: {best_config['best_eta']}")
    print(f"  Best Mu: {best_config['best_mu']}")
    print(f"  Best Grad Type: {best_config['best_grad_type']}")
    
    # Train final model with best hyperparameters
    final_model = EquilibriumMatchingModel(
        target_dim=target_dim,
        cond_dim=0,
        source_sampler=gaussian_sampler(target_dim),
        lambda_=best_config['lambda_'],
        gamma_type=best_config['gamma_type'],
        objective_type=best_config['objective_type'],
        energy_type=best_config['energy_type']
    )
    
    # Train for longer with best config
    print("Training final model with best hyperparameters...")
    final_history = train_flow_model(
        flow_model=final_model,
        train_data=train_data,
        val_data=val_data,
        epochs=1000,  # Longer training for final model
        batch_size=batch_size,
        learning_rate=best_config['learning_rate'],
        device=device,
        verbose=True
    )
    
    # Generate samples for visualization using best inference parameters
    print("Generating samples for visualization...")
    with torch.no_grad():
        generated_samples = final_model.predict(
            batch_size=1000,
            device=device,
            num_steps=500,
            eta=best_config['best_eta'],
            mu=best_config['best_mu'],
            grad_type=best_config['best_grad_type']
        )
    
    # Plot comparison
    print("Creating comparison plots...")
    plot_data_comparison(
        target_data=test_data,
        generated_data=generated_samples,
        save_path=os.path.join(plots_dir, 'equilibrium_best_config_comparison.png')
    )
    
    # Plot training history
    plot_training_history(final_history, save_path=os.path.join(plots_dir, 'equilibrium_best_config_training.png'))
    
    print("Hyperparameter sweep completed successfully!")
    print(f"All results saved in '{plots_dir}' directory")


def run_single_model_inference_experiment():
    """
    Train a single EquilibriumMatchingModel and roll it out on all provided inference parameters.
    Saves a plot for each inference parameter combination in plots_dir/inference_plots/.
    Rolls out twice for each inference param to show variability.
    """
    print("Starting Single Model Inference Experiment...")
    
    # Create plots directory structure
    plots_dir = "test_plots"
    inference_plots_dir = os.path.join(plots_dir, "inference_plots")
    os.makedirs(inference_plots_dir, exist_ok=True)
    print(f"Created plots directory: {plots_dir}")
    print(f"Created inference plots directory: {inference_plots_dir}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate bimodal 2D data
    print("Generating bimodal 2D data...")
    target_data = generate_bimodal_data(n_samples=5000, noise=0.1)
    print(f"Generated {target_data.shape[0]} samples")
    
    # Split data into train/validation/test
    n_samples = target_data.shape[0]
    indices = torch.randperm(n_samples)
    
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = target_data[train_indices]
    val_data = target_data[val_indices]
    test_data = target_data[test_indices]
    
    print(f"Train samples: {train_data.shape[0]}")
    print(f"Validation samples: {val_data.shape[0]}")
    print(f"Test samples: {test_data.shape[0]}")
    
    # Define inference parameter grid
    eta_values = np.linspace(0.003, 0.3, 20)
    mu_values = [0.035, 0.35]
    grad_types = ['nag']
    
    print(f"Inference parameter grid:")
    print(f"  Eta values: {eta_values}")
    print(f"  Mu values: {mu_values}")
    print(f"  Grad types: {grad_types}")
    print(f"  Total combinations: {len(eta_values) * len(mu_values) * len(grad_types)}")
    
    # Training configuration
    epochs = 1000
    batch_size = 256
    target_dim = 2
    learning_rate = 1e-4
    
    # Create and train the model
    print(f"\n{'='*60}")
    print("Training Single Model")
    print(f"{'='*60}")
    
    flow_model = EquilibriumMatchingModel(
        target_dim=target_dim,
        cond_dim=0,
        source_sampler=gaussian_sampler(target_dim),
        lambda_=1.0,
        gamma_type='linear',
        objective_type='implicit',
        energy_type='dot'
    )
    
    print("Training the model...")
    history = train_flow_model(
        flow_model=flow_model,
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        verbose=True
    )
    
    print(f"Training completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
    # Test all inference parameter combinations
    print(f"\n{'='*60}")
    print("Testing Inference Parameters")
    print(f"{'='*60}")
    
    rollout_count = 0
    total_rollouts = len(eta_values) * len(mu_values) * len(grad_types) * 2  # 2 rollouts per combination
    
    for eta in eta_values:
        for mu in mu_values:
            for grad_type in grad_types:
                print(f"\nTesting eta={eta}, mu={mu}, grad_type={grad_type}...")
                
                # Roll out twice for each parameter combination
                for rollout_idx in range(2):
                    rollout_count += 1
                    print(f"  Rollout {rollout_idx + 1}/2 ({rollout_count}/{total_rollouts})...")
                    
                    # Generate samples with current inference parameters
                    with torch.no_grad():
                        generated_samples = flow_model.predict(
                            batch_size=1000,
                            device=device,
                            num_steps=500,
                            eta=eta,
                            mu=mu,
                            grad_type=grad_type
                        )
                    
                    # Create filename for this combination
                    filename = f"inference_eta_{eta}_mu_{mu}_grad_{grad_type}_rollout_{rollout_idx + 1}.png"
                    save_path = os.path.join(inference_plots_dir, filename)
                    
                    # Create comparison plot
                    plot_data_comparison(
                        target_data=test_data,
                        generated_data=generated_samples,
                        save_path=save_path,
                        title=f"Inference: η={eta}, μ={mu}, {grad_type.upper()} (Rollout {rollout_idx + 1})"
                    )
                    
                    print(f"    Saved plot: {filename}")
    
    # Create a summary plot showing all inference parameters
    print(f"\nCreating summary plot...")
    create_inference_summary_plot(
        flow_model=flow_model,
        target_data=test_data,
        eta_values=eta_values,
        mu_values=mu_values,
        grad_types=grad_types,
        device=device,
        save_path=os.path.join(inference_plots_dir, "inference_summary.png")
    )
    
    # Save training history plot
    plot_training_history(
        history, 
        save_path=os.path.join(plots_dir, 'single_model_training_history.png')
    )
    
    print(f"\n{'='*60}")
    print("SINGLE MODEL INFERENCE EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"Total rollouts: {rollout_count}")
    print(f"Plots saved in: {inference_plots_dir}")
    print(f"Training history saved in: {plots_dir}")


def create_inference_summary_plot(flow_model, target_data, eta_values, mu_values, grad_types, device, save_path):
    """
    Create a summary plot showing results for all inference parameter combinations.
    """
    n_etas = len(eta_values)
    n_mus = len(mu_values)
    n_grads = len(grad_types)
    
    # Create subplot grid
    fig, axes = plt.subplots(n_grads, n_etas * n_mus, figsize=(4 * n_etas * n_mus, 4 * n_grads))
    if n_grads == 1:
        axes = axes.reshape(1, -1)
    if n_etas * n_mus == 1:
        axes = axes.reshape(-1, 1)
    
    for grad_idx, grad_type in enumerate(grad_types):
        for eta_idx, eta in enumerate(eta_values):
            for mu_idx, mu in enumerate(mu_values):
                col_idx = eta_idx * n_mus + mu_idx
                ax = axes[grad_idx, col_idx]
                
                # Generate samples
                with torch.no_grad():
                    generated_samples = flow_model.predict(
                        batch_size=500,  # Smaller batch for summary
                        device=device,
                        num_steps=500,
                        eta=eta,
                        mu=mu,
                        grad_type=grad_type
                    )
                
                # Plot target data
                ax.scatter(target_data[:, 0].cpu().numpy(), target_data[:, 1].cpu().numpy(), 
                          alpha=0.3, s=1, c='blue', label='Target')
                
                # Plot generated data
                ax.scatter(generated_samples[:, 0].cpu().numpy(), generated_samples[:, 1].cpu().numpy(), 
                          alpha=0.5, s=1, c='red', label='Generated')
                
                ax.set_title(f'η={eta}, μ={mu}, {grad_type.upper()}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved: {save_path}")


def run_action_chunk_steering_experiment():
    """
    Action chunk steering experiment using trajectory data.
    
    - Uses action chunks of dimension 16 from trajectory data
    - Human actions serve as prior/condition for predicting robot actions
    - Two modes: Mode 1 (robot_mode_1, human_mode_1) and Mode 2 (robot_mode_2, human_mode_2)
    - Train conditional flow models to predict robot actions given human actions
    - Test steering by providing human action chunks as conditions
    """
    print("Starting Action Chunk Steering Experiment...")
    
    # Import the trajectory generation functions
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'test'))
    from prior_trajectory_steering import generate_paired_action_chunks
    
    plots_dir = "test_plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Created plots directory: {plots_dir}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate trajectory data and paired action chunks
    print("Generating trajectory data and action chunks...")
    pairs_data = generate_paired_action_chunks(n_trajectories=200, n_steps=128, chunk_size=16)
    
    # Extract robot and human action chunks
    mode_1_pairs = pairs_data['mode_1_pairs']
    mode_2_pairs = pairs_data['mode_2_pairs']
    
    print(f"Generated {len(mode_1_pairs)} mode 1 pairs")
    print(f"Generated {len(mode_2_pairs)} mode 2 pairs")
    
    # Convert to tensors - extract actions (y coordinates) and states (first x coordinate)
    def chunks_to_actions_and_states(pairs_list):
        robot_actions = []
        human_actions = []
        robot_states = []
        human_states = []
        for pair in pairs_list:
            # Actions are the y coordinates (16-dimensional)
            robot_actions.append(torch.tensor(pair['robot'][:, 1], dtype=torch.float32))
            human_actions.append(torch.tensor(pair['human'][:, 1], dtype=torch.float32))
            # States are the first x coordinate (1-dimensional)
            robot_states.append(torch.tensor(pair['robot'][0, 0], dtype=torch.float32))
            human_states.append(torch.tensor(pair['human'][0, 0], dtype=torch.float32))
        return (torch.stack(robot_actions), torch.stack(human_actions), 
                torch.stack(robot_states), torch.stack(human_states))
    
    # Mode 1 data
    robot_mode_1, human_mode_1, robot_state_1, human_state_1 = chunks_to_actions_and_states(mode_1_pairs)
    # Mode 2 data  
    robot_mode_2, human_mode_2, robot_state_2, human_state_2 = chunks_to_actions_and_states(mode_2_pairs)
    
    print(f"Robot mode 1 actions shape: {robot_mode_1.shape}")
    print(f"Human mode 1 actions shape: {human_mode_1.shape}")
    print(f"Robot mode 1 states shape: {robot_state_1.shape}")
    print(f"Human mode 1 states shape: {human_state_1.shape}")
    print(f"Robot mode 2 actions shape: {robot_mode_2.shape}")
    print(f"Human mode 2 actions shape: {human_mode_2.shape}")
    print(f"Robot mode 2 states shape: {robot_state_2.shape}")
    print(f"Human mode 2 states shape: {human_state_2.shape}")
    
    # Combine data for training
    all_robot_actions = torch.cat([robot_mode_1, robot_mode_2], dim=0)
    all_human_actions = torch.cat([human_mode_1, human_mode_2], dim=0)
    all_robot_states = torch.cat([robot_state_1, robot_state_2], dim=0)
    all_human_states = torch.cat([human_state_1, human_state_2], dim=0)
    
    # Create mode labels (0 for mode 1, 1 for mode 2)
    mode_labels = torch.cat([
        torch.zeros(len(mode_1_pairs), dtype=torch.long),
        torch.ones(len(mode_2_pairs), dtype=torch.long)
    ])
    
    # Split data into train/val/test (60/20/20)
    n_samples = all_robot_actions.shape[0]
    indices = torch.randperm(n_samples)
    
    train_n = int(0.6 * n_samples)
    val_n = int(0.2 * n_samples)
    
    train_idx = indices[:train_n]
    val_idx = indices[train_n:train_n + val_n]
    test_idx = indices[train_n + val_n:]
    
    train_robot = all_robot_actions[train_idx]
    train_human = all_human_actions[train_idx]
    train_robot_state = all_robot_states[train_idx]
    train_human_state = all_human_states[train_idx]
    train_modes = mode_labels[train_idx]
    
    val_robot = all_robot_actions[val_idx]
    val_human = all_human_actions[val_idx]
    val_robot_state = all_robot_states[val_idx]
    val_human_state = all_human_states[val_idx]
    val_modes = mode_labels[val_idx]
    
    test_robot = all_robot_actions[test_idx]
    test_human = all_human_actions[test_idx]
    test_robot_state = all_robot_states[test_idx]
    test_human_state = all_human_states[test_idx]
    test_modes = mode_labels[test_idx]
    
    print(f"Train samples: {train_robot.shape[0]}")
    print(f"Validation samples: {val_robot.shape[0]}")
    print(f"Test samples: {test_robot.shape[0]}")
    
    # Create source sampler for robot actions (Gaussian)
    action_dim = 16  # 16 y-coordinates
    state_dim = 1    # 1 x-coordinate (first state)
    source_sampler = gaussian_sampler(action_dim, mean=0.0, std=1.0)
    
    # Create conditional flow model (human actions + states as condition)
    print("Creating conditional flow model...")
    flow_model = ConditionalFlowModel(
        target_dim=action_dim,  # Robot action dimension (16 y-coordinates)
        cond_dim=action_dim + state_dim,  # Human action (16) + human state (1) as condition
        source_sampler=source_sampler
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    
    # Training configuration
    epochs = 500
    batch_size = 128
    learning_rate = 1e-3
    
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Combine human actions and states as conditions
    train_conditions = torch.cat([train_human, train_human_state.unsqueeze(1)], dim=1)
    val_conditions = torch.cat([val_human, val_human_state.unsqueeze(1)], dim=1)
    
    print(f"Train conditions shape: {train_conditions.shape}")
    print(f"Val conditions shape: {val_conditions.shape}")
    
    # Train the model
    print("Training the action chunk steering model...")
    history = train_flow_model(
        flow_model=flow_model,
        train_data=train_robot,
        val_data=val_robot,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        verbose=True,
        train_conditions=train_conditions,
        val_conditions=val_conditions
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history, save_path=os.path.join(plots_dir, 'action_chunk_training_history.png'))
    
    # Test steering with different human action conditions
    print("Testing action chunk steering...")
    
    # Test 1: Generate robot actions conditioned on mode 1 human actions and states
    with torch.no_grad():
        # Sample some mode 1 human actions and states as conditions
        mode_1_test_indices = torch.where(test_modes == 0)[0][:500]
        mode_1_human_actions = test_human[mode_1_test_indices]
        mode_1_human_states = test_human_state[mode_1_test_indices]
        mode_1_robot_targets = test_robot[mode_1_test_indices]
        
        # Combine human actions and states as conditions
        mode_1_conditions = torch.cat([mode_1_human_actions, mode_1_human_states.unsqueeze(1)], dim=1)
        
        # Generate robot actions conditioned on mode 1 human actions and states
        mode_1_generated = flow_model.predict(
            batch_size=len(mode_1_conditions),
            condition=mode_1_conditions,
            device=device,
            num_steps=100
        )
    
    # Test 2: Generate robot actions conditioned on mode 2 human actions and states
    with torch.no_grad():
        # Sample some mode 2 human actions and states as conditions
        mode_2_test_indices = torch.where(test_modes == 1)[0][:500]
        mode_2_human_actions = test_human[mode_2_test_indices]
        mode_2_human_states = test_human_state[mode_2_test_indices]
        mode_2_robot_targets = test_robot[mode_2_test_indices]
        
        # Combine human actions and states as conditions
        mode_2_conditions = torch.cat([mode_2_human_actions, mode_2_human_states.unsqueeze(1)], dim=1)
        
        # Generate robot actions conditioned on mode 2 human actions and states
        mode_2_generated = flow_model.predict(
            batch_size=len(mode_2_conditions),
            condition=mode_2_conditions,
            device=device,
            num_steps=100
        )
    
    # Visualize results by reshaping back to trajectory chunks
    def visualize_action_chunks(robot_targets, robot_generated, human_actions, human_states, mode_name, save_path):
        """Visualize action chunks by reshaping back to 16x2 trajectories"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Action Chunk Steering - {mode_name}', fontsize=16)
        
        # Show first 5 examples
        n_examples = min(5, len(robot_targets))
        
        for i in range(n_examples):
            # Create x coordinates (assuming uniform spacing from 0 to 1)
            x_coords = torch.linspace(0, 1, 16)
            
            # Robot trajectories: x_coords + robot actions (y coordinates)
            target_traj = torch.stack([x_coords, robot_targets[i]], dim=1).cpu().numpy()
            generated_traj = torch.stack([x_coords, robot_generated[i]], dim=1).cpu().numpy()
            
            # Human trajectory: x_coords + human actions (y coordinates), starting from human state
            human_x_start = human_states[i].item()
            human_x_coords = torch.linspace(human_x_start, human_x_start + 1, 16)
            human_traj = torch.stack([human_x_coords, human_actions[i]], dim=1).cpu().numpy()
            
            # Plot target vs generated
            axes[0, i].plot(target_traj[:, 0], target_traj[:, 1], 'o-', 
                           linewidth=2, markersize=4, color='blue', alpha=0.7, label='Target Robot')
            axes[0, i].plot(generated_traj[:, 0], generated_traj[:, 1], 's-', 
                           linewidth=2, markersize=4, color='red', alpha=0.7, label='Generated Robot')
            axes[0, i].set_title(f'Example {i+1}: Robot Actions')
            axes[0, i].set_xlabel('x')
            axes[0, i].set_ylabel('y')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend()
            
            # Plot human condition
            axes[1, i].plot(human_traj[:, 0], human_traj[:, 1], '^-', 
                           linewidth=2, markersize=4, color='green', alpha=0.7, label='Human Condition')
            axes[1, i].set_title(f'Example {i+1}: Human Condition (start x={human_x_start:.2f})')
            axes[1, i].set_xlabel('x')
            axes[1, i].set_ylabel('y')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_action_chunks(
        mode_1_robot_targets, mode_1_generated, mode_1_human_actions, mode_1_human_states,
        'Mode 1', os.path.join(plots_dir, 'action_chunk_steering_mode1.png')
    )
    
    visualize_action_chunks(
        mode_2_robot_targets, mode_2_generated, mode_2_human_actions, mode_2_human_states,
        'Mode 2', os.path.join(plots_dir, 'action_chunk_steering_mode2.png')
    )
    
    # Evaluate performance
    print("Evaluating action chunk steering performance...")
    
    # Calculate MSE for each mode
    mode_1_mse = torch.mean((mode_1_generated - mode_1_robot_targets) ** 2).item()
    mode_2_mse = torch.mean((mode_2_generated - mode_2_robot_targets) ** 2).item()
    
    print(f"Mode 1 MSE: {mode_1_mse:.6f}")
    print(f"Mode 2 MSE: {mode_2_mse:.6f}")
    
    # Test cross-mode steering (use mode 1 human to generate mode 2 robot)
    print("Testing cross-mode steering...")
    with torch.no_grad():
        # Use mode 1 human actions and states to condition mode 2 robot generation
        cross_mode_conditions = torch.cat([mode_1_human_actions, mode_1_human_states.unsqueeze(1)], dim=1)
        cross_mode_generated = flow_model.predict(
            batch_size=len(cross_mode_conditions),
            condition=cross_mode_conditions,
            device=device,
            num_steps=100
        )
    
    # Visualize cross-mode steering
    visualize_action_chunks(
        mode_2_robot_targets[:len(cross_mode_generated)], cross_mode_generated, mode_1_human_actions, mode_1_human_states,
        'Cross-Mode (Mode 1 Human → Mode 2 Robot)', os.path.join(plots_dir, 'action_chunk_cross_mode_steering.png')
    )
    
    cross_mode_mse = torch.mean((cross_mode_generated - mode_2_robot_targets[:len(cross_mode_generated)]) ** 2).item()
    print(f"Cross-mode MSE: {cross_mode_mse:.6f}")
    
    print("Action chunk steering experiment completed successfully!")
    print(f"All plots saved in '{plots_dir}' directory:")
    print(f"  - action_chunk_training_history.png")
    print(f"  - action_chunk_steering_mode1.png")
    print(f"  - action_chunk_steering_mode2.png")
    print(f"  - action_chunk_cross_mode_steering.png")


if __name__ == "__main__":
    # #### EXPERIMENT OPTIONS
    run_experiment(model_type='fm', epochs=500)
    #run_experiment(model_type='eqm', epochs=100)
    #run_conditional_experiment()
    # run_single_model_inference_experiment()
    # run_action_chunk_steering_experiment()

    #run_conditional_experiment()
    #run_equilibrium_hyperparameter_sweep()
    #run_single_model_inference_experiment()
    # run_spiral_experiment()
    # run_source_comparison_experiment()
    # run_prior_steering_experiment()
    # run_prior_vs_paired_simple_experiment()
