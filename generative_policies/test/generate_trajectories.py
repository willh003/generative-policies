import numpy as np
import matplotlib.pyplot as plt

# Very low frequency base trajectory for p1 (linear-ish)
# Very low frequency base trajectory for p1 (sinusoidal - modality A)
def base_traj_p1(x):
    return 0.3 * (np.sin(2 * np.pi * x) - np.sin(0))

# Very low frequency base trajectory for p2 (sinusoidal with slight variation - modality A)
def base_traj_p2(x):
    return 0.3 * (np.sin(2 * np.pi * x + np.pi/6) - np.sin(np.pi/6))

# Very low frequency base trajectory for p3 (different sinusoidal - modality B)
def base_traj_p3(x):
    return 0.35 * (np.sin(3 * np.pi * x) - np.sin(0))

# Very low frequency base trajectory for p4 (similar to p3 - modality B)
def base_traj_p4(x):
    return 0.35 * (np.sin(3 * np.pi * x + np.pi/6) - np.sin(np.pi/6))


def generate_trajectory_distribution(n_trajectories, n_steps, base_trajectory_func, noise_scale=0.02, high_freq_range=(10, 30), seed=None):
    """
    Generate tightly clustered trajectories around a base function.
    """
    if seed is not None:
        np.random.seed(seed)
    
    trajectories = np.zeros((n_trajectories, n_steps, 2))
    
    # Generate x coordinates (shared across all trajectories)
    x_coords = np.linspace(0, 1, n_steps)
    
    for i in range(n_trajectories):
        # X coordinates are the same for all trajectories
        trajectories[i, :, 0] = x_coords
        
        # Y follows the base trajectory with small high-freq noise
        y_base = base_trajectory_func(x_coords)
        
        # Add small high-frequency noise unique to this trajectory
        high_freq_noise = noise_scale * np.sin(np.random.uniform(high_freq_range[0], high_freq_range[1]) * np.pi * x_coords + np.random.uniform(0, 2*np.pi))
        high_freq_noise += noise_scale * 0.5 * np.sin(np.random.uniform(high_freq_range[1], high_freq_range[1]+20) * np.pi * x_coords + np.random.uniform(0, 2*np.pi))
        
        # Add small random offset for starting y position
        y_offset = np.random.uniform(-0.05, 0.05)
        
        trajectories[i, :, 1] = y_base + y_offset + high_freq_noise
    
    return trajectories


def get_bimodal_trajectories(n_trajectories, n_steps):
    # Generate trajectories with different high-frequency features
    robot_mode_1 = generate_trajectory_distribution(n_trajectories, n_steps, base_traj_p1, noise_scale=0.02, high_freq_range=(10, 25), seed=42)
    human_mode_1 = generate_trajectory_distribution(n_trajectories, n_steps, base_traj_p2, noise_scale=0.02, high_freq_range=(40, 60), seed=43)
    robot_mode_2 = generate_trajectory_distribution(n_trajectories, n_steps, base_traj_p3, noise_scale=0.02, high_freq_range=(10, 25), seed=44)
    human_mode_2 = generate_trajectory_distribution(n_trajectories, n_steps, base_traj_p4, noise_scale=0.02, high_freq_range=(40, 60), seed=45)
    return robot_mode_1, human_mode_1, robot_mode_2, human_mode_2

if __name__ == "__main__":
    n_trajectories = 100
    n_steps = 256

    robot_mode_1, human_mode_1, robot_mode_2, human_mode_2 = get_bimodal_trajectories(n_trajectories, n_steps)

    # Plot all together
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = ['darkblue', 'lightblue', 'darkred', 'lightcoral']
    for idx, (trajs, base_func, label, color) in enumerate([
        (robot_mode_1, base_traj_p1, 'Modality A, Robot', colors[0]),
        (human_mode_1, base_traj_p2, 'Modality A, Human', colors[1]),
        (robot_mode_2, base_traj_p3, 'Modality B, Robot', colors[2]),
        (human_mode_2, base_traj_p4, 'Modality B, Human', colors[3])
    ]):
        for i in range(min(15, n_trajectories)):
            ax.plot(trajs[i, :, 0], trajs[i, :, 1], alpha=0.3, linewidth=0.8, color=color, 
                    label=label if i==0 else '')
        x_line = np.linspace(0, 1, 200)
        ax.plot(x_line, base_func(x_line), '--', linewidth=2, color=color)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Two modalities with different high-frequency features')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig('test_plots/trajectories.png')
    plt.close()