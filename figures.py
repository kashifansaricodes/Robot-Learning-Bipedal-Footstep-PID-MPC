import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)
iterations = np.arange(1, 101)  # 100 training iterations on x-axis

# 1. Average Episode Reward (higher is better)
# RL + PD learns fast initially but plateaus lower; RL + MPC learns slower but ends higher and steadier
reward_pd = 80 * (1 - np.exp(-iterations/20.0))  # asymptote ~80
reward_mpc = 95 * (1 - np.exp(-iterations/40.0))  # asymptote ~95 (slower rise)
# Add noise to simulate variability (PD has more variance than MPC)
reward_pd += np.random.normal(0, 5, size=iterations.shape)
reward_mpc += np.random.normal(0, 3, size=iterations.shape)
# Ensure rewards stay non-negative
reward_pd = np.clip(reward_pd, 0, None)
reward_mpc = np.clip(reward_mpc, 0, None)

# 2. Joint Stability Index (JSI, lower is better)
# Start higher and decrease over time; PD starts less stable (higher JSI) than MPC
jsi_pd = 2 + (6-2) * np.exp(-iterations/30.0)   # starts ~6, goes to ~2
jsi_mpc = 1 + (5-1) * np.exp(-iterations/50.0)  # starts ~5, goes to ~1.6
jsi_pd += np.random.normal(0, 0.3, size=iterations.shape)   # PD more noise (oscillations)
jsi_mpc += np.random.normal(0, 0.1, size=iterations.shape)  # MPC very stable
jsi_pd = np.clip(jsi_pd, 0, None)
jsi_mpc = np.clip(jsi_mpc, 0, None)

# 3. Torque Usage (Mean torque magnitude, in arbitrary units)
# Both decrease as control improves; RL+PD uses higher torque on average than RL+MPC
torque_pd = 50 + (80-50) * np.exp(-iterations/50.0)  # starts ~80, drops toward ~50
torque_mpc = 40 + (70-40) * np.exp(-iterations/60.0) # starts ~70, drops toward ~40
torque_pd += np.random.normal(0, 2, size=iterations.shape)   # PD more variation
torque_mpc += np.random.normal(0, 1, size=iterations.shape)  # MPC smoother
torque_pd = np.clip(torque_pd, 0, None)
torque_mpc = np.clip(torque_mpc, 0, None)

# 4. Training Pace (Steps per second, higher means faster training)
# RL+PD can run training faster (simple control), RL+MPC is slower due to computation
pace_pd = np.linspace(70, 100, len(iterations))  # linearly from 70 to 100 steps/sec
pace_mpc = np.linspace(50, 80, len(iterations))  # linearly from 50 to 80 steps/sec
pace_pd += np.random.normal(0, 2, size=iterations.shape)
pace_mpc += np.random.normal(0, 2, size=iterations.shape)
pace_pd = np.clip(pace_pd, 0, None)
pace_mpc = np.clip(pace_mpc, 0, None)

# 5. Average Episode Length (steps per episode, higher means robot stays up longer)
# Simulate RL+PD achieving high lengths quickly but with some drop, RL+MPC steadily increasing
key_iters = np.array([1, 20, 40, 60, 100])
pd_lengths = np.array([50, 120, 180, 160, 190])   # RL+PD: quick rise, slight dip, then high
mpc_lengths = np.array([100, 120, 140, 160, 180]) # RL+MPC: starts higher initially, steady rise
episode_pd = np.interp(iterations, key_iters, pd_lengths)
episode_mpc = np.interp(iterations, key_iters, mpc_lengths)
episode_pd += np.random.normal(0, 5, size=iterations.shape)
episode_mpc += np.random.normal(0, 3, size=iterations.shape)
episode_pd = np.clip(episode_pd, 0, 200)
episode_mpc = np.clip(episode_mpc, 0, 200)

# Plotting the metrics on subplots
fig, axs = plt.subplots(5, 1, figsize=(8, 14), sharex=True)
# Define a list of metrics to loop through for plotting
metrics = [
    (reward_pd, reward_mpc, "Average Episode Reward", "Reward"),
    (jsi_pd, jsi_mpc, "Joint Stability Index (Lower = Better)", "Stability Index"),
    (torque_pd, torque_mpc, "Torque Usage (Mean Torque)", "Torque (arb. units)"),
    (pace_pd, pace_mpc, "Training Pace (Steps/sec)", "Steps per Second"),
    (episode_pd, episode_mpc, "Average Episode Length", "Steps per Episode")
]
# Plot each metric curve
for ax, (data_pd, data_mpc, title, y_label) in zip(axs, metrics):
    ax.plot(iterations, data_pd, '--o', label="RL + PD", markevery=10)
    ax.plot(iterations, data_mpc, '-^', label="RL + MPC", markevery=10)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    ax.grid(True, linestyle='--', alpha=0.7)
# Set common x-axis label
axs[-1].set_xlabel("Training Iterations")
# Add an overall title for the figure
fig.suptitle("RL + PD vs RL + MPC Training Metrics (Simulated)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
