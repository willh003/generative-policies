import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path to import modules
from .prior_trajectory_steering import generate_paired_action_chunks
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from generative_policies.action_translation.flow_action_translator import (
    FlowActionConditionedTranslator, FlowActionPriorTranslator, FlowActionPriorConditionedTranslator
)



def run_action_translator_comparison_experiment():
    """
    Compare FlowActionConditionedTranslator vs FlowActionPriorTranslator using trajectory data.
    
    - Uses action chunks of dimension 16 from trajectory data
    - Human actions serve as action_prior for both translators
    - Robot actions are the target actions to predict
    - States (first x coordinate) serve as observations
    - Compare two approaches:
      1) FlowActionConditionedTranslator: learns p(robot_action | state, human_action)
      2) FlowActionPriorTranslator: learns p(robot_action | state) using human_action as prior
    """
    print("Starting Action Translator Comparison Experiment...")
    
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

        robot_cur_states = []
        human_cur_states = []
        for pair in pairs_list:
            # Actions are the y coordinates (16-dimensional)
            robot_actions.append(torch.tensor(pair['robot'][:, 1], dtype=torch.float32))
            human_actions.append(torch.tensor(pair['human'][:, 1], dtype=torch.float32))
            # States are the first x coordinate (1-dimensional)
            robot_states.append(torch.tensor(pair['robot'][:, 0], dtype=torch.float32))
            human_states.append(torch.tensor(pair['human'][:, 0], dtype=torch.float32))
            robot_cur_states.append(torch.tensor(pair['robot'][0, 0], dtype=torch.float32))
            human_cur_states.append(torch.tensor(pair['human'][0, 0], dtype=torch.float32))
        return (torch.stack(robot_actions), torch.stack(human_actions), 
                torch.stack(robot_states), torch.stack(human_states),
                torch.stack(robot_cur_states), torch.stack(human_cur_states))
    
    # Mode 1 data
    robot_mode_1, human_mode_1, robot_state_1, human_state_1, robot_cur_state_1, human_cur_state_1 = chunks_to_actions_and_states(mode_1_pairs)
    # Mode 2 data  
    robot_mode_2, human_mode_2, robot_state_2, human_state_2, robot_cur_state_2, human_cur_state_2 = chunks_to_actions_and_states(mode_2_pairs)
    
    print(f"Robot mode 1 actions shape: {robot_mode_1.shape}")
    print(f"Human mode 1 actions shape: {human_mode_1.shape}")
    print(f"Robot mode 1 states shape: {robot_state_1.shape}")
    print(f"Human mode 1 states shape: {human_state_1.shape}")
    print(f"Robot mode 1 current states shape: {robot_cur_state_1.shape}")
    print(f"Human mode 1 current states shape: {human_cur_state_1.shape}")
    # Combine data for training
    all_robot_actions = torch.cat([robot_mode_1, robot_mode_2], dim=0)
    all_human_actions = torch.cat([human_mode_1, human_mode_2], dim=0)
    all_robot_states = torch.cat([robot_state_1, robot_state_2], dim=0)
    all_human_states = torch.cat([human_state_1, human_state_2], dim=0)
    all_robot_cur_states = torch.cat([robot_cur_state_1, robot_cur_state_2], dim=0)
    all_human_cur_states = torch.cat([human_cur_state_1, human_cur_state_2], dim=0)
    
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
    
    train_robot_actions = all_robot_actions[train_idx]
    train_human_actions = all_human_actions[train_idx]
    train_robot_states = all_robot_states[train_idx]
    train_human_states = all_human_states[train_idx]
    train_robot_cur_states = all_robot_cur_states[train_idx]
    train_human_cur_states = all_human_cur_states[train_idx]
    train_modes = mode_labels[train_idx]
    
    val_robot_actions = all_robot_actions[val_idx]
    val_human_actions = all_human_actions[val_idx]
    val_robot_states = all_robot_states[val_idx]
    val_human_states = all_human_states[val_idx]
    val_modes = mode_labels[val_idx]
    val_robot_cur_states = all_robot_cur_states[val_idx]
    val_human_cur_states = all_human_cur_states[val_idx]
    
    test_robot_actions = all_robot_actions[test_idx]
    test_human_actions = all_human_actions[test_idx]
    test_robot_states = all_robot_states[test_idx]
    test_human_states = all_human_states[test_idx]
    test_robot_cur_states = all_robot_cur_states[test_idx]
    test_human_cur_states = all_human_cur_states[test_idx]
    test_modes = mode_labels[test_idx]
    
    print(f"Train samples: {train_robot_actions.shape[0]}")
    print(f"Validation samples: {val_robot_actions.shape[0]}")
    print(f"Test samples: {test_robot_actions.shape[0]}")
    
    # Define model parameters
    action_dim = 16  # 16 y-coordinates
    obs_dim = 1      # 1 x-coordinate (first state)
    
    # Create both translators
    print("Creating action translators...")
    conditioned_translator = FlowActionPriorConditionedTranslator(
        action_dim=action_dim,
        obs_dim=obs_dim,
        device=device
    )
    
    prior_translator = FlowActionPriorTranslator(
        action_dim=action_dim,
        obs_dim=obs_dim,
        device=device
    )
    
    print(f"Conditioned translator parameters: {sum(p.numel() for p in conditioned_translator.parameters()):,}")
    print(f"Prior translator parameters: {sum(p.numel() for p in prior_translator.parameters()):,}")
    
    # Training configuration
    epochs = 700
    batch_size = 128
    learning_rate = 1e-3
    
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Training function for action translators
    def train_translator(translator, train_obs, train_action_prior, train_action, 
                        val_obs, val_action_prior, val_action, epochs, batch_size, lr, device):
        optimizer = torch.optim.Adam(translator.parameters(), lr=lr)
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            translator.train()
            train_loss = 0.0
            n_train_batches = 0
            
            for i in range(0, len(train_obs), batch_size):
                batch_obs = train_obs[i:i+batch_size].unsqueeze(1).to(device)  # Add dimension for obs_dim
                batch_action_prior = train_action_prior[i:i+batch_size].to(device)
                batch_action = train_action[i:i+batch_size].to(device)
                
                optimizer.zero_grad()
                loss = translator(batch_obs, batch_action_prior, batch_action)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_train_batches += 1
            
            # Validation
            translator.eval()
            val_loss = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for i in range(0, len(val_obs), batch_size):
                    batch_obs = val_obs[i:i+batch_size].unsqueeze(1).to(device)  # Add dimension for obs_dim
                    batch_action_prior = val_action_prior[i:i+batch_size].to(device)
                    batch_action = val_action[i:i+batch_size].to(device)
                    
                    loss = translator(batch_obs, batch_action_prior, batch_action)
                    val_loss += loss.item()
                    n_val_batches += 1
            
            avg_train_loss = train_loss / n_train_batches
            avg_val_loss = val_loss / n_val_batches
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        return {'train_loss': train_losses, 'val_loss': val_losses}
    
    # Train conditioned translator
        # Train prior translator
    print("Training prior translator...")
    prior_history = train_translator(
        prior_translator, train_robot_cur_states, train_human_actions, train_robot_actions,
        val_robot_cur_states, val_human_actions, val_robot_actions,
        epochs, batch_size, learning_rate, device
    )
    print("Training conditioned translator...")
    conditioned_history = train_translator(
        conditioned_translator, train_robot_cur_states, train_human_actions, train_robot_actions,
        val_robot_cur_states, val_human_actions, val_robot_actions,
        epochs, batch_size, learning_rate, device
    )
    

    
    # Plot training histories
    print("Plotting training histories...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(conditioned_history['train_loss'], label='Conditioned Train', color='blue')
    plt.plot(conditioned_history['val_loss'], label='Conditioned Val', color='blue', linestyle='--')
    plt.plot(prior_history['train_loss'], label='Prior Train', color='red')
    plt.plot(prior_history['val_loss'], label='Prior Val', color='red', linestyle='--')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(conditioned_history['val_loss'], label='Conditioned', color='blue')
    plt.plot(prior_history['val_loss'], label='Prior', color='red')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'translator_training_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test both translators
    print("Testing action translators...")
    
    # Test on mode 1 data
    mode_1_test_indices = torch.where(test_modes == 0)[0][:100]
    mode_1_obs = test_robot_cur_states[mode_1_test_indices].unsqueeze(1)  # Add dimension for obs_dim
    mode_1_robot_states = test_robot_states[mode_1_test_indices]
    mode_1_action_prior = test_human_actions[mode_1_test_indices]
    mode_1_target = test_robot_actions[mode_1_test_indices]
    mode_1_human_states = test_human_states[mode_1_test_indices]

    # different action but same state
    assert torch.allclose(mode_1_robot_states, mode_1_human_states)
        
    
    # Test on mode 2 data
    mode_2_test_indices = torch.where(test_modes == 1)[0][:100]
    mode_2_obs = test_robot_cur_states[mode_2_test_indices].unsqueeze(1)  # Add dimension for obs_dim
    mode_2_robot_states = test_robot_states[mode_2_test_indices]
    mode_2_action_prior = test_human_actions[mode_2_test_indices]
    mode_2_target = test_robot_actions[mode_2_test_indices]
    mode_2_human_states = test_human_states[mode_2_test_indices]
    assert torch.allclose(mode_2_robot_states, mode_2_human_states)

    
    # Generate predictions
    with torch.no_grad():
        # Mode 1 predictions
        mode_1_conditioned_pred = conditioned_translator.predict(
            mode_1_obs.cpu().numpy(), mode_1_action_prior.cpu().numpy()
        )
        mode_1_prior_pred = prior_translator.predict(
            mode_1_obs.cpu().numpy(), mode_1_action_prior.cpu().numpy()
        )
        
        # Mode 2 predictions
        mode_2_conditioned_pred = conditioned_translator.predict(
            mode_2_obs.cpu().numpy(), mode_2_action_prior.cpu().numpy()
        )
        mode_2_prior_pred = prior_translator.predict(
            mode_2_obs.cpu().numpy(), mode_2_action_prior.cpu().numpy()
        )
    
    # Convert back to tensors for evaluation
    mode_1_conditioned_pred = torch.tensor(mode_1_conditioned_pred, dtype=torch.float32)
    mode_1_prior_pred = torch.tensor(mode_1_prior_pred, dtype=torch.float32)
    mode_2_conditioned_pred = torch.tensor(mode_2_conditioned_pred, dtype=torch.float32)
    mode_2_prior_pred = torch.tensor(mode_2_prior_pred, dtype=torch.float32)
    
    # Calculate MSE for each mode and translator
    mode_1_conditioned_mse = torch.mean((mode_1_conditioned_pred - mode_1_target) ** 2).item()
    mode_1_prior_mse = torch.mean((mode_1_prior_pred - mode_1_target) ** 2).item()
    mode_2_conditioned_mse = torch.mean((mode_2_conditioned_pred - mode_2_target) ** 2).item()
    mode_2_prior_mse = torch.mean((mode_2_prior_pred - mode_2_target) ** 2).item()
    
    print(f"Mode 1 - Conditioned MSE: {mode_1_conditioned_mse:.6f}")
    print(f"Mode 1 - Prior MSE: {mode_1_prior_mse:.6f}")
    print(f"Mode 2 - Conditioned MSE: {mode_2_conditioned_mse:.6f}")
    print(f"Mode 2 - Prior MSE: {mode_2_prior_mse:.6f}")
    
    # Visualization function
    def visualize_translator_predictions(targets, conditioned_pred, prior_pred, obs, all_states, action_prior, mode_name, save_path):
        """Visualize translator predictions"""
        # Show first 5 examples
        n_examples = min(5, len(targets))
        fig, axes = plt.subplots(2, n_examples, figsize=(3*n_examples, 10))
        fig.suptitle(f'Action Translator Comparison - {mode_name}', fontsize=16)
        
        for i in range(n_examples):
            x_coords = all_states[i]
            # Target trajectory
            target_traj = torch.stack([x_coords, targets[i]], dim=1).cpu().numpy()
            
            # Conditioned prediction
            conditioned_traj = torch.stack([x_coords, conditioned_pred[i]], dim=1).cpu().numpy()
            
            # Prior prediction
            prior_traj = torch.stack([x_coords, prior_pred[i]], dim=1).cpu().numpy()
            
            # Action prior (human action) - use the same x coordinates as the main trajectory
            action_prior_traj = torch.stack([x_coords, action_prior[i]], dim=1).cpu().numpy()
            
            # Handle both single and multiple examples
            if n_examples == 1:
                ax_top = axes[0]
                ax_bottom = axes[1]
            else:
                ax_top = axes[0, i]
                ax_bottom = axes[1, i]
            
            # Plot target vs predictions
            ax_top.plot(target_traj[:, 0], target_traj[:, 1], 'o-', 
                       linewidth=2, markersize=4, color='blue', alpha=0.7, label='Target')
            ax_top.plot(conditioned_traj[:, 0], conditioned_traj[:, 1], 's-', 
                       linewidth=2, markersize=4, color='red', alpha=0.7, label='Conditioned')
            ax_top.plot(prior_traj[:, 0], prior_traj[:, 1], '^-', 
                       linewidth=2, markersize=4, color='green', alpha=0.7, label='Prior')
            ax_top.set_title(f'Example {i+1}: Predictions')
            ax_top.set_xlabel('x')
            ax_top.set_ylabel('y')
            ax_top.grid(True, alpha=0.3)
            ax_top.legend()
            
            # Plot action prior
            ax_bottom.plot(action_prior_traj[:, 0], action_prior_traj[:, 1], 'd-', 
                          linewidth=2, markersize=4, color='orange', alpha=0.7, label='Action Prior')
            ax_bottom.set_title(f'Example {i+1}: Action Prior (obs x={obs[i].item():.2f})')
            ax_bottom.set_xlabel('x')
            ax_bottom.set_ylabel('y')
            ax_bottom.grid(True, alpha=0.3)
            ax_bottom.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create visualizations
    print("Creating visualizations...")
    
    visualize_translator_predictions(
        mode_1_target, mode_1_conditioned_pred, mode_1_prior_pred, 
        mode_1_obs, mode_1_robot_states, mode_1_action_prior,
        'Mode 1', os.path.join(plots_dir, 'translator_comparison_mode1.png')   
    )
    
    visualize_translator_predictions(
        mode_2_target, mode_2_conditioned_pred, mode_2_prior_pred,
        mode_2_obs, mode_2_robot_states, mode_2_action_prior,
        'Mode 2', os.path.join(plots_dir, 'translator_comparison_mode2.png')
    )
    
    # Test cross-mode steering
    print("Testing cross-mode steering...")
    with torch.no_grad():
        # Use mode 1 human actions to predict mode 2 robot actions
        cross_mode_conditioned_pred = conditioned_translator.predict(
            mode_2_obs.cpu().numpy(), mode_1_action_prior.cpu().numpy()
        )
        cross_mode_prior_pred = prior_translator.predict(
            mode_2_obs.cpu().numpy(), mode_1_action_prior.cpu().numpy()
        )
    
    cross_mode_conditioned_pred = torch.tensor(cross_mode_conditioned_pred, dtype=torch.float32)
    cross_mode_prior_pred = torch.tensor(cross_mode_prior_pred, dtype=torch.float32)
    
    cross_mode_conditioned_mse = torch.mean((cross_mode_conditioned_pred - mode_2_target) ** 2).item()
    cross_mode_prior_mse = torch.mean((cross_mode_prior_pred - mode_2_target) ** 2).item()
    
    print(f"Cross-mode - Conditioned MSE: {cross_mode_conditioned_mse:.6f}")
    print(f"Cross-mode - Prior MSE: {cross_mode_prior_mse:.6f}")
    
    # Visualize cross-mode steering
    visualize_translator_predictions(
        mode_2_target, cross_mode_conditioned_pred, cross_mode_prior_pred,
        mode_2_obs, mode_2_robot_states, mode_1_action_prior,
        'Cross-Mode (Mode 1 Human → Mode 2 Robot)', os.path.join(plots_dir, 'translator_cross_mode_comparison.png')
    )
    
    # Generate entire trajectories using flow models
    print("Generating entire trajectories...")
    
    def generate_full_trajectory(translator, test_trajectory, human_trajectory, mode_name, save_path):
        """Generate a full trajectory by running the flow model on each x coordinate"""
        x_coords = test_trajectory[:, 0]  # All x coordinates
        y_target = test_trajectory[:, 1]  # Target y coordinates
        
        # Generate predictions for each x coordinate
        predicted_actions = []
        with torch.no_grad():
            for i in range(len(x_coords)):
                # Current observation (x coordinate)
                obs = torch.tensor([[x_coords[i]]], dtype=torch.float32)
                # Human action at this step - need to create a 16-dimensional action chunk
                # For simplicity, repeat the single y coordinate to create a 16-dim vector
                human_action = torch.full((16,), human_trajectory[i, 1], dtype=torch.float32)
                
                # Get prediction from flow model
                pred = translator.predict(obs.cpu().numpy(), human_action.cpu().numpy())
                predicted_actions.append(pred[0][0])  # Take first element of first prediction
        
        predicted_actions = np.array(predicted_actions)
        
        # Create visualization
        fig, (ax1) = plt.subplots(1, 1, figsize=(6, 10))
        fig.suptitle(f'Full Trajectory Generation - {mode_name}', fontsize=16)
        
        # Plot 1: Target vs Predicted trajectory
        ax1.plot(x_coords, y_target, 'o-', linewidth=2, markersize=4, color='blue', alpha=0.7, label='Target')
        ax1.plot(x_coords, predicted_actions, 's-', linewidth=2, markersize=4, color='red', alpha=0.7, label='Predicted')
        ax1.plot(human_trajectory[:, 0], human_trajectory[:, 1], 'd-', linewidth=2, markersize=4, color='orange', alpha=0.7, label='Human Steering')
        ax1.set_title('Target vs Predicted Trajectory')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate MSE for the full trajectory
        mse = np.mean((predicted_actions - y_target) ** 2)
        print(f"{mode_name} Full Trajectory MSE: {mse:.6f}")
        
        return predicted_actions, mse
    
    # Get a full test trajectory for each mode
    # Use the first trajectory from the original data
    test_traj_1 = pairs_data['original_trajectories']['robot_mode_1'][0]  # First robot trajectory
    human_traj_1 = pairs_data['original_trajectories']['human_mode_1'][0]  # First human trajectory
    
    test_traj_2 = pairs_data['original_trajectories']['robot_mode_2'][0]  # First robot trajectory
    human_traj_2 = pairs_data['original_trajectories']['human_mode_2'][0]  # First human trajectory
    
    # Generate full trajectories for both modes and both translators
    print("Generating full trajectory for Mode 1...")
    mode_1_conditioned_pred_full, mode_1_conditioned_mse_full = generate_full_trajectory(
        conditioned_translator, test_traj_1, human_traj_1, 
        'Mode 1 - Conditioned', os.path.join(plots_dir, 'full_trajectory_mode1_conditioned.png')
    )
    
    mode_1_prior_pred_full, mode_1_prior_mse_full = generate_full_trajectory(
        prior_translator, test_traj_1, human_traj_1, 
        'Mode 1 - Prior', os.path.join(plots_dir, 'full_trajectory_mode1_prior.png')
    )
    
    print("Generating full trajectory for Mode 2...")
    mode_2_conditioned_pred_full, mode_2_conditioned_mse_full = generate_full_trajectory(
        conditioned_translator, test_traj_2, human_traj_2, 
        'Mode 2 - Conditioned', os.path.join(plots_dir, 'full_trajectory_mode2_conditioned.png')
    )
    
    mode_2_prior_pred_full, mode_2_prior_mse_full = generate_full_trajectory(
        prior_translator, test_traj_2, human_traj_2, 
        'Mode 2 - Prior', os.path.join(plots_dir, 'full_trajectory_mode2_prior.png')
    )
    
    print("Action translator comparison experiment completed successfully!")
    print(f"All plots saved in '{plots_dir}' directory:")
    print(f"  - translator_training_comparison.png")
    print(f"  - translator_comparison_mode1.png")
    print(f"  - translator_comparison_mode2.png")
    print(f"  - translator_cross_mode_comparison.png")
    print(f"  - full_trajectory_mode1_conditioned.png")
    print(f"  - full_trajectory_mode1_prior.png")
    print(f"  - full_trajectory_mode2_conditioned.png")
    print(f"  - full_trajectory_mode2_prior.png")


if __name__ == "__main__":
    run_action_translator_comparison_experiment()

