import numpy as np
import matplotlib.pyplot as plt
from .generate_trajectories import get_bimodal_trajectories

def generate_paired_action_chunks(n_trajectories=100, n_steps=256, chunk_size=16):
    """
    Generate trajectories and break them into paired action chunks.
    
    Args:
        n_trajectories: Number of trajectories to generate
        n_steps: Number of steps per trajectory
        chunk_size: Size of each action chunk
    
    Returns:
        Dictionary containing paired chunks for both modes
    """
    # Generate trajectories using the existing function
    robot_mode_1, human_mode_1, robot_mode_2, human_mode_2 = get_bimodal_trajectories(n_trajectories, n_steps)
    
    # Create paired chunks for mode 1 (robot_mode_1 with human_mode_1)
    mode_1_pairs = []
    for k in range(n_steps - chunk_size + 1):
        for traj_idx in range(n_trajectories):
            robot_chunk = robot_mode_1[traj_idx, k:k+chunk_size, :]
            human_chunk = human_mode_1[traj_idx, k:k+chunk_size, :]
            mode_1_pairs.append({
                'robot': robot_chunk,
                'human': human_chunk,
                'start_step': k,
                'trajectory_idx': traj_idx
            })
    
    # Create paired chunks for mode 2 (robot_mode_2 with human_mode_2)
    mode_2_pairs = []
    for k in range(n_steps - chunk_size + 1):
        for traj_idx in range(n_trajectories):
            robot_chunk = robot_mode_2[traj_idx, k:k+chunk_size, :]
            human_chunk = human_mode_2[traj_idx, k:k+chunk_size, :]
            mode_2_pairs.append({
                'robot': robot_chunk,
                'human': human_chunk,
                'start_step': k,
                'trajectory_idx': traj_idx
            })
    
    return {
        'mode_1_pairs': mode_1_pairs,
        'mode_2_pairs': mode_2_pairs,
        'original_trajectories': {
            'robot_mode_1': robot_mode_1,
            'human_mode_1': human_mode_1,
            'robot_mode_2': robot_mode_2,
            'human_mode_2': human_mode_2
        }
    }

def visualize_paired_sequences(pairs_data, n_examples=5, chunk_size=16):
    """
    Visualize paired action chunks.
    
    Args:
        pairs_data: Dictionary containing paired chunks
        n_examples: Number of example pairs to show
        chunk_size: Size of each chunk
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Paired Action Chunks Visualization', fontsize=16)
    
    # Mode 1 visualization
    ax1 = axes[0, 0]  # Robot mode 1
    ax2 = axes[0, 1]  # Human mode 1
    
    # Mode 2 visualization  
    ax3 = axes[1, 0]  # Robot mode 2
    ax4 = axes[1, 1]  # Human mode 2
    
    # Sample random pairs for visualization
    indices = np.random.choice(min(len(pairs_data['mode_1_pairs']), len(pairs_data['mode_2_pairs'])), n_examples, replace=False)
    
    # Plot mode 1 pairs
    for i, idx in enumerate(indices):
        pair = pairs_data['mode_1_pairs'][idx]
        alpha = 0.7 - (i * 0.1)  # Varying transparency
        if alpha < .1:
            alpha = .1
        
        # Robot chunk
        ax3.plot(pair['robot'][:, 0], pair['robot'][:, 1], 
                alpha=alpha, linewidth=2, color='darkblue',
                label=f'Robot Chunk {i+1}' if i < 3 else '')
        
        # Human chunk
        ax1.plot(pair['human'][:, 0], pair['human'][:, 1], 
                alpha=alpha, linewidth=2, color='lightblue',
                label=f'Human Chunk {i+1}' if i < 3 else '')
    
    # Plot mode 2 pairs
    for i, idx in enumerate(indices):
        pair = pairs_data['mode_2_pairs'][idx]
        alpha = 0.7 - (i * 0.1)  # Varying transparency
        if alpha < .1:
            alpha = .1
        # Robot chunk
        ax4.plot(pair['robot'][:, 0], pair['robot'][:, 1], 
                alpha=alpha, linewidth=2, color='darkred',
                label=f'Robot Chunk {i+1}' if i < 3 else '')
        
        # Human chunk
        ax2.plot(pair['human'][:, 0], pair['human'][:, 1], 
                alpha=alpha, linewidth=2, color='lightcoral',
                label=f'Human Chunk {i+1}' if i < 3 else '')
    
    # Set labels and titles
    ax1.set_title('Mode 1 - Robot Chunks')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    
    ax2.set_title('Mode 1 - Human Chunks')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    
    ax3.set_title('Mode 2 - Robot Chunks')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_xlim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    
    ax4.set_title('Mode 2 - Human Chunks')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_xlim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    plt.savefig('test_plots/paired_action_chunks.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_corresponding_pairs(pairs_data, n_examples=3):
    """
    Visualize corresponding robot-human pairs side by side.
    
    Args:
        pairs_data: Dictionary containing paired chunks
        n_examples: Number of example pairs to show
    """
    fig, axes = plt.subplots(n_examples, 2, figsize=(12, 4*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Corresponding Robot-Human Pairs', fontsize=16)
    
    # Sample random pairs for both modes
    mode_1_indices = np.random.choice(len(pairs_data['mode_1_pairs']), n_examples, replace=False)
    mode_2_indices = np.random.choice(len(pairs_data['mode_2_pairs']), n_examples, replace=False)
    
    for i in range(n_examples):
        # Mode 1 pair
        pair1 = pairs_data['mode_1_pairs'][mode_1_indices[i]]
        axes[i, 0].plot(pair1['robot'][:, 0], pair1['robot'][:, 1], 
                       'o-', linewidth=2, markersize=4, color='darkblue', alpha=0.8,
                       label='Robot')
        axes[i, 0].plot(pair1['human'][:, 0], pair1['human'][:, 1], 
                       's-', linewidth=2, markersize=4, color='lightblue', alpha=0.8,
                       label='Human')
        axes[i, 0].set_title(f'Mode 1 - Step {pair1["start_step"]}, Traj {pair1["trajectory_idx"]}')
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend()
        
        # Mode 2 pair
        pair2 = pairs_data['mode_2_pairs'][mode_2_indices[i]]
        axes[i, 1].plot(pair2['robot'][:, 0], pair2['robot'][:, 1], 
                       'o-', linewidth=2, markersize=4, color='darkred', alpha=0.8,
                       label='Robot')
        axes[i, 1].plot(pair2['human'][:, 0], pair2['human'][:, 1], 
                       's-', linewidth=2, markersize=4, color='lightcoral', alpha=0.8,
                       label='Human')
        axes[i, 1].set_title(f'Mode 2 - Step {pair2["start_step"]}, Traj {pair2["trajectory_idx"]}')
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel('y')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend()
    
    plt.tight_layout()
    plt.savefig('test_plots/corresponding_pairs.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Generate trajectories and paired chunks
    print("Generating trajectories and paired action chunks...")
    pairs_data = generate_paired_action_chunks(n_trajectories=100, n_steps=128, chunk_size=16)
    
    print(f"Generated {len(pairs_data['mode_1_pairs'])} mode 1 pairs")
    print(f"Generated {len(pairs_data['mode_2_pairs'])} mode 2 pairs")
    
    # Visualize paired sequences
    print("Creating visualizations...")
    visualize_paired_sequences(pairs_data, n_examples=8)
    visualize_corresponding_pairs(pairs_data, n_examples=3)
    
    print("Visualization complete! Check 'paired_action_chunks.png' and 'corresponding_pairs.png'")
