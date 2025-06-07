import os
import sys
import time
from collections import deque
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from baselines.common import explained_variance
from baselines.common import tf_util as U
import ppo.policies as policies
from ppo.policies import MLPCNNPolicy

from rlenv.cassie_env import CassieEnv
from configs.defaults import ROOT_PATH
from configs.env_config import config_mpc_train
from baselines import logger
from mpi4py import MPI
import argparse

# set numpy print options
np.set_printoptions(precision=3, suppress=True)

os.environ["OPENAI_LOG_FORMAT"] = "stdout,log,tensorboard"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_folder = ROOT_PATH + "/ckpts/"


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0
    
    myseed = i + 1000 * rank if i is not None else None
    tf.set_random_seed(myseed)
    np.random.seed(myseed)


def get_args():
    parser = argparse.ArgumentParser(description="training_setup")

    parser.add_argument(
        "--train_name", type=str, default="mpc_integrated", help="naming the training"
    )

    parser.add_argument(
        "--rnd_seed", type=int, default=42, help="random seed (default 42)"
    )

    parser.add_argument(
        "--max_iters", type=int, default=5000, help="max iterations (default 5000)"
    )

    parser.add_argument(
        "--restore_from",
        type=str,
        default=None,
        help="restore_from previous checkpoint (default None)",
    )

    parser.add_argument(
        "--restore_cont",
        type=int,
        default=None,
        help="counts that has been resumed (default None)",
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="save the ckpt every save_interval iteration",
    )
    
    parser.add_argument(
        "--use_mpc",
        type=bool,
        default=True,
        help="use MPC during training (default True)",
    )
    
    parser.add_argument(
        "--mpc_weight",
        type=float,
        default=0.1,
        help="weight for MPC integration (default 0.1)",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate (default 1e-4)",
    )
    
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.98,
        help="discount factor (default 0.98)",
    )
    
    parser.add_argument(
        "--lam",
        type=float,
        default=0.95,
        help="GAE lambda (default 0.95)",
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable MuJoCo visualization during training",
    )

    args = parser.parse_args()

    return args


args = get_args()

# name for training
train_name = args.train_name  
# if the training is restored from a previous ckpt, None for from scratch
restore_from = args.restore_from  
# how many time has the training been stopped and contiuned
restore_cont = args.restore_cont
# random seed of the trianing
rnd_seed = args.rnd_seed 
# max iteration for training 
max_iters = args.max_iters
# how frequent the ckpt will be saved
save_interval = args.save_interval
# whether to use MPC
use_mpc = args.use_mpc
# MPC integration weight
mpc_weight = args.mpc_weight
# Learning rate
lr = args.lr
# Discount factor
gamma = args.gamma
# GAE lambda
lam = args.lam
# Whether to enable MuJoCo visualization
visualize = args.visualize

# define the name
if restore_cont and restore_from:
    saved_model = train_name + "_rnds" + str(rnd_seed) + "_cont" + str(restore_cont)
else:
    saved_model = train_name + "_rnds" + str(rnd_seed)

restore_model_from_file = restore_from
os.environ["OPENAI_LOGDIR"] = ROOT_PATH + "/logs/" + saved_model

print("[Train]: MODEL_TO_SAVE", saved_model)
print(f"Training with{'out' if not use_mpc else ''} MPC integration")
print(f"MPC integration weight: {mpc_weight}")


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # Last element is 0 for last element
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def safe_policy_act(policy, stochastic, ob_vf, ob_pol):
    """Safely call policy.act with error handling"""
    try:
        # Make sure ob_pol is a tuple containing both parts needed for the CNN
        if not isinstance(ob_pol, tuple) or len(ob_pol) != 2:
            print(f"Warning: ob_pol not in expected format, got {type(ob_pol)}")
            # Return reasonable defaults
            action_shape = policy.ac_space.shape[0] if hasattr(policy, 'ac_space') else 10
            return np.zeros(action_shape), 0.0
            
        return policy.act(stochastic, ob_vf, ob_pol)
    except Exception as e:
        print(f"Error in policy.act: {e}")
        # Return reasonable defaults
        action_shape = policy.ac_space.shape[0] if hasattr(policy, 'ac_space') else 10
        return np.zeros(action_shape), 0.0


def traj_segment_generator(env, policy, timesteps_per_batch, stochastic, mpc_weight_param=0.1, start_episode=0, iteration=0):
    """
    Generate trajectories for training
    """
    # Initialize buffer
    t = 0
    ac = np.zeros(env.action_space.shape[0], dtype=np.float32)
    
    # Setup buffers for storing trajectories
    ob_vf, ob_pol = env.reset()
    ob_vf_dim = ob_vf.shape[0]
    
    # Check whether ob_pol is in the expected format (should be a tuple of two arrays)
    if not isinstance(ob_pol, tuple) or len(ob_pol) != 2:
        print(f"Warning: Environment returned ob_pol in unexpected format: {type(ob_pol)}")
        print("Attempting to reshape observation to expected format")
        # Try to recover by creating a default observation format
        if not isinstance(ob_pol, tuple):
            # Create a tuple with two parts based on environment space definitions
            ob_pol_0 = np.zeros(env.observation_space_pol.shape) if hasattr(env, 'observation_space_pol') else np.zeros((10, len(env.sim2pol_indices) if hasattr(env, 'sim2pol_indices') else 20))
            ob_pol_1 = np.zeros(env.observation_space_pol_cnn.shape) if hasattr(env, 'observation_space_pol_cnn') else np.zeros((10, 10, 1))
            ob_pol = (ob_pol_0, ob_pol_1)
            print(f"Created default ob_pol with shapes: {ob_pol[0].shape}, {ob_pol[1].shape}")
    else:
        print(f"Got valid ob_pol with shapes: {ob_pol[0].shape}, {ob_pol[1].shape}")
        
    # Make sure we capture both parts of ob_pol for passing to the policy
    # The CNN part should be shaped as [37, 60, 1] as expected by the network
    ob_pol_base = ob_pol[0] if isinstance(ob_pol, tuple) and len(ob_pol) > 0 else np.zeros((10, 20))
    # Create the correctly shaped CNN input
    if isinstance(ob_pol, tuple) and len(ob_pol) > 1:
        ob_pol_cnn_shape = ob_pol[1].shape
        # Check if the shape matches what the network expects
        if ob_pol_cnn_shape != (37, 60, 1):
            print(f"Reshaping ob_pol_cnn from {ob_pol_cnn_shape} to (37, 60, 1)")
            ob_pol_cnn = np.zeros((37, 60, 1))
        else:
            ob_pol_cnn = ob_pol[1]
    else:
        print("Creating default ob_pol_cnn with shape (37, 60, 1)")
        ob_pol_cnn = np.zeros((37, 60, 1))
    
    # Indices for MPC suggestions and stability metrics in observations
    # The last dimensions should be: [mpc_suggestions (10) + stability_metrics (3)]
    expected_mpc_size = env.num_motor + 3  # 10 motor suggestions + 3 stability metrics
    
    # If observation has MPC data appended (observation is longer than without MPC)
    if len(ob_vf) > 100:  # Arbitrary threshold, adjust based on your observation size
        # Verify there's enough space for MPC data
        if len(ob_vf) >= expected_mpc_size:
            mpc_start_idx = len(ob_vf) - expected_mpc_size
            stability_metrics_idx = len(ob_vf) - 3
            has_mpc_data = True
        else:
            has_mpc_data = False
            mpc_start_idx = 0
            stability_metrics_idx = 0
    else:
        # If the environment doesn't include MPC data, use defaults
        has_mpc_data = False
        mpc_start_idx = 0
        stability_metrics_idx = 0
        
    # Get dimensions for ac_pred and vpred
    stochastic_ac, vpred_val = safe_policy_act(policy, stochastic, ob_vf, ob_pol)
    
    # Buffers
    new = True
    rew = 0.0
    ob_vfs = np.array([ob_vf for _ in range(timesteps_per_batch)])
    acs = np.array([ac for _ in range(timesteps_per_batch)])
    vpreds = np.zeros(timesteps_per_batch, 'float32')
    news = np.zeros(timesteps_per_batch, 'int32')
    rews = np.zeros(timesteps_per_batch, 'float32')
    
    # Important: Initialize these as lists to track episode-level metrics
    ep_rews = []  # episode rewards - will be populated with FULL episode rewards
    ep_lens = []  # episode lengths - will be populated with FULL episode lengths
    
    # Track current episode data
    current_ep_rew = 0.0  # Track reward for the current episode
    current_ep_len = 0    # Track length for the current episode
    
    # Add buffers for MPC suggestions and stability metrics
    mpc_acs = np.zeros((timesteps_per_batch, env.num_motor), dtype=np.float32)
    stability_metrics = np.zeros((timesteps_per_batch, 3), dtype=np.float32)
    
    # Check if visualization is enabled
    is_visual = hasattr(env, 'is_visual') and env.is_visual
    
    # Initialize episode counter with provided start value
    episode_count = start_episode
    current_iteration = iteration
    
    # Loop through and collect trajectories
    total_steps_collected = 0  # Track total steps collected in this iteration
    steps_in_this_ep = 0  # Track steps in current episode
    
    while True:
        # Save observations
        ob_vfs[t] = ob_vf
        vpreds[t] = vpred_val
        news[t] = new
        
        # Extract MPC suggestions and stability metrics from observations
        if has_mpc_data and len(ob_vf) >= mpc_start_idx + env.num_motor and len(ob_vf) >= stability_metrics_idx + 3:
            try:
                # Extract MPC suggestions if they exist in the observation
                mpc_acs[t] = ob_vf[mpc_start_idx:mpc_start_idx+env.num_motor]
                stability_metrics[t] = ob_vf[stability_metrics_idx:stability_metrics_idx+3]
            except Exception as e:
                mpc_acs[t] = np.zeros(env.num_motor)
                stability_metrics[t] = np.zeros(3)
        else:
            mpc_acs[t] = np.zeros(env.num_motor)
            stability_metrics[t] = np.zeros(3)
        
        # Take action
        ac, vpred_val = safe_policy_act(policy, stochastic, ob_vf, ob_pol)
        
        # Blend MPC suggestion with policy action when needed based on stability
        roll_pitch_tilt = np.abs(stability_metrics[t][:2])  # roll and pitch tilt
        z_vel = stability_metrics[t][2]  # z velocity
        
        # Determine if we need MPC help
        needs_stabilization = False
        if has_mpc_data:  # Only attempt stabilization if MPC data is available
            if np.any(roll_pitch_tilt > 0.15) or z_vel < -0.2:  # Thresholds for intervention
                needs_stabilization = True
        
        # Apply MPC suggestion with adaptive weight based on stability
        if needs_stabilization:
            # Calculate severity (how unstable the robot is)
            severity = max(
                np.max(roll_pitch_tilt) / 0.2,  # Normalize by threshold
                abs(min(z_vel, 0)) / 0.3       # Normalize by threshold
            )
            # Cap severity at 1.0
            severity = min(severity, 1.0)
            
            # Apply weighted combination of policy action and MPC suggestion
            mpc_blend_weight = mpc_weight_param * severity
            ac = (1 - mpc_blend_weight) * ac + mpc_blend_weight * mpc_acs[t]
        
        acs[t] = ac
        
        # Step environment
        ob_vf, ob_pol, rew, new, info = env.step(ac)
        rews[t] = rew
        
        # Accumulate reward for the current episode
        current_ep_rew += rew
        current_ep_len += 1
        
        # Check if robot has fallen but 'new' flag wasn't set
        # Force reset if robot seems to have fallen but episode wasn't marked as done
        if not new and hasattr(env, 'fall_flag') and env.fall_flag:
            new = True
            ob_vf, ob_pol = env.reset()
        
        # If visualization is enabled, render the environment
        if is_visual:
            env.render()
            
            # Check for pause in visualization
            if hasattr(env, 'vis') and hasattr(env.vis, 'ispaused') and env.vis.ispaused():
                print("Visualization paused, press 'p' to resume")
                while env.vis.ispaused():
                    env.render()
        
        # Increment counters
        t += 1
        steps_in_this_ep += 1
        total_steps_collected += 1
        
        # Handle episode completion
        if new:
            episode_count += 1
            
            # Save episode stats
            ep_rews.append(current_ep_rew)  # Save total episode reward
            ep_lens.append(current_ep_len)  # Save total episode length
            
            # Reset episode counters
            current_ep_rew = 0.0
            current_ep_len = 0
            
            # Calculate progress toward completing this iteration
            progress_pct = (total_steps_collected / timesteps_per_batch) * 100
            
            # Display episode info with iteration progress
            print(f"Episode {episode_count} complete: length={steps_in_this_ep}, reward={sum(rews[max(0, t-steps_in_this_ep):t]):.3f} | " +
                  f"Iteration {current_iteration+1} progress: {progress_pct:.1f}% ({total_steps_collected}/{timesteps_per_batch} steps)")
            
            steps_in_this_ep = 0
            
            # Reset environment for next episode
            ob_vf, ob_pol = env.reset()
            
            # Safety check: if observation contains NaN values, reset environment again
            if np.isnan(ob_vf).any() or (isinstance(ob_pol, tuple) and (np.isnan(ob_pol[0]).any() or np.isnan(ob_pol[1]).any())):
                print("Warning: NaN detected in observation after reset. Resetting environment again.")
                ob_vf, ob_pol = env.reset()
        
        # Check if we have enough transitions - either from current buffer or total collected
        if t >= timesteps_per_batch or total_steps_collected >= timesteps_per_batch:
            # Add final return value to the last state
            _, vpred_val = safe_policy_act(policy, stochastic, ob_vf, ob_pol)
            
            # Cut buffers to actual size
            ob_vfs = ob_vfs[:t]
            acs = acs[:t]
            vpreds = vpreds[:t]
            news = news[:t]
            rews = rews[:t]
            mpc_acs = mpc_acs[:t]
            stability_metrics = stability_metrics[:t]
            
            # Return trajectory data with episode rewards and lengths
            # Ensure both ep_rews and ep_lens are non-empty by adding dummy values if needed
            if len(ep_rews) == 0:
                print("Warning: No complete episodes in this batch. Adding a dummy episode.")
                ep_rews = [0.0]  # Default value
                ep_lens = [0]    # Default value
            
            # Return trajectory data - ensure both ep_rew and ep_rews keys exist for compatibility
            yield {
                "ob_vf": ob_vfs,
                "ob_pol": ob_pol_base,
                "ob_pol_cnn": ob_pol_cnn,
                "ac": acs,
                "vpred": vpreds,
                "nextvpred": vpred_val,
                "new": news,
                "rew": rews,
                "ep_rew": ep_rews,   # Use consistent naming - this will be used in learn_with_mpc
                "ep_rews": ep_rews,  # Also include with alternate name for backward compatibility
                "ep_lens": ep_lens,
                "mpc_acs": mpc_acs,
                "stability_metrics": stability_metrics
            }
            
            # Reset counters and buffers
            t = 0
            total_steps_collected = 0
            steps_in_this_ep = 0
            ob_vfs = np.array([ob_vf for _ in range(timesteps_per_batch)])
            acs = np.array([ac for _ in range(timesteps_per_batch)])
            vpreds = np.zeros(timesteps_per_batch, 'float32')
            news = np.zeros(timesteps_per_batch, 'int32')
            rews = np.zeros(timesteps_per_batch, 'float32')
            mpc_acs = np.zeros((timesteps_per_batch, env.num_motor), dtype=np.float32)
            stability_metrics = np.zeros((timesteps_per_batch, 3), dtype=np.float32)
            
            # Don't reset episode tracking between batches
            # This allows us to properly track episodes that span batch boundaries
            ep_rews = []
            ep_lens = []


def learn_with_mpc(
    max_timesteps=1e6,
    eval_freq=5e3,
    save_freq=5e3,
    seed=42,
    lr=1.5e-4,
    gamma=0.995,
    lam=0.95,
    optim_stepsize=1.5e-4,
    optim_epochs=3,
    optim_batchsize=512,
    clip_param=0.2,
    entcoeff=0.0,
    vf_coeff=0.5,
    max_grad_norm=0.5,
    mpc_integration_weight_param=0.1,  # Parameter to control how much to blend MPC suggestions
    visualize=False  # Flag to enable/disable visualization
):
    """
    Train an agent using PPO with MPC integration on the Cassie robot
    """
    # Define the timesteps per batch for each iteration (used by the generator and for progress reporting)
    timesteps_per_batch = 2048
    
    # 1. Create Environment
    # Configure environment to use MPC
    config = config_mpc_train.copy()
    config["use_mpc"] = True
    config["mpc_horizon"] = 10
    config["mpc_dt"] = 0.01
    config["max_timesteps"] = 5000
    config["is_visual"] = visualize  # Enable/disable visualization
    config["cam_track_robot"] = visualize  # Track the robot with camera when visualizing
    
    print(f"Visualization {'enabled' if visualize else 'disabled'} during training")
    env = CassieEnv(config=config)
    
    # 2. Build PPO Model
    sess = U.make_session()
    sess.__enter__()
    
    # Get dimensions from environment
    ob_space_pol = env.observation_space_pol
    ob_space_vf = env.observation_space_vf
    ob_space_pol_cnn = env.observation_space_pol_cnn
    ac_space = env.action_space
    
    # Setup policy model
    pi = policies.MLPCNNPolicy(
        name="pi",
        ob_space_vf=ob_space_vf,
        ob_space_pol=ob_space_pol,
        ob_space_pol_cnn=ob_space_pol_cnn,
        ac_space=ac_space, 
        hid_size=512,
        num_hid_layers=2
    )
    
    # Create old policy for PPO
    oldpi = policies.MLPCNNPolicy(
        name="oldpi",
        ob_space_vf=ob_space_vf,
        ob_space_pol=ob_space_pol,
        ob_space_pol_cnn=ob_space_pol_cnn,
        ac_space=ac_space, 
        hid_size=512,
        num_hid_layers=2
    )
    
    cliprange = tf.placeholder(dtype=tf.float32, shape=[])
    mpc_weight = tf.placeholder(dtype=tf.float32, shape=[])  # Control MPC integration weight
    
    vf_coef = vf_coeff
    entcoeff = entcoeff
    
    # Create PPO loss function
    ob_vf = U.get_placeholder_cached(name="ob_vf")
    ob_pol = U.get_placeholder_cached(name="ob_pol")
    ob_pol_cnn = U.get_placeholder_cached(name="ob_pol_cnn")  # Make sure this placeholder is properly defined
    ac = pi.pdtype.sample_placeholder([None])
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return
    mpc_suggested_ac = tf.placeholder(dtype=tf.float32, shape=[None, env.num_motor])  # MPC suggested action
    
    # Extract MPC suggestions from observations (last 13 elements: 10 motor commands + 3 stability metrics)
    # Assuming observations include MPC suggestions at the end
    mpc_suggestion_idx = -13  # Position where MPC suggestions start in observation
    
    # Get PPO action distribution and values
    ac_pred = pi.pd.mean
    vpred = pi.vpred
    
    # Create loss function that integrates MPC suggestions
    old_vpred = tf.placeholder(dtype=tf.float32, shape=[None])
    old_ac_pred = tf.placeholder(dtype=tf.float32, shape=[None, ac_space.shape[0]])
    
    # Regular PPO loss calculation
    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
    atarg_clip = ratio * atarg
    atarg_clip_surr1 = ratio * atarg
    atarg_clip_surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
    surr_loss = -tf.reduce_mean(tf.minimum(atarg_clip_surr1, atarg_clip_surr2))
    vf_loss = tf.reduce_mean(tf.square(vpred - ret))
    entloss = -tf.reduce_mean(pi.pd.entropy())
    pol_entpen = entcoeff * entloss
    vf_loss_final = vf_coef * vf_loss
    pol_surr_loss = surr_loss
    
    # Add MPC integration loss - encourages the policy to consider MPC suggestions
    # Calculate distance between policy action and MPC suggestion
    mpc_integration_loss = tf.reduce_mean(tf.square(ac_pred - mpc_suggested_ac))
    # This term encourages the policy to move toward MPC suggestions when stability metrics indicate it's needed
    mpc_weighted_loss = mpc_weight * mpc_integration_loss
    
    # Final loss integrates MPC suggestions with adaptive weight
    total_loss = pol_surr_loss + pol_entpen + vf_loss_final + mpc_weighted_loss
    
    losses = [pol_surr_loss, pol_entpen, vf_loss_final, total_loss, mpc_weighted_loss]
    loss_names = ["pol_surr_loss", "pol_entpen", "vf_loss_final", "total_loss", "mpc_loss"]
    
    # Setup optimizer
    params = tf.trainable_variables()
    grads = tf.gradients(total_loss, params)
    if max_grad_norm is not None:
        grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    grads = list(zip(grads, params))
    
    trainer = tf.train.AdamOptimizer(learning_rate=optim_stepsize, epsilon=1e-5)
    _train = trainer.apply_gradients(grads)
    
    # Initialize and sync params
    tf.global_variables_initializer().run()
    
    # Create assigners to update the old policy params
    # Fix: Manually ensure variable alignment between the models instead of automatic assignment
    pi_vars = pi.get_variables()
    oldpi_vars = oldpi.get_variables()
    
    # Debug information to track variable shapes
    print("Pi and Oldpi variable shapes:")
    for i, (pv, opv) in enumerate(zip(pi_vars, oldpi_vars)):
        print(f"Var {i}: Pi {pv.shape}, Oldpi {opv.shape}")
    
    # Create update operations only for variables with matching shapes
    update_ops = []
    for oldv, newv in zip(oldpi_vars, pi_vars):
        if oldv.shape == newv.shape:
            update_ops.append(tf.assign(oldv, newv))
        else:
            print(f"WARNING: Skipping variable with incompatible shapes: {oldv.shape} vs {newv.shape}")
    
    assign_old_eq_new = U.function([], [], updates=update_ops)
    
    # Function to compute losses and do optimization
    # This is the problematic part with shape mismatches
    compute_losses = U.function(
        [ob_vf, ob_pol, ob_pol_cnn, ac, atarg, ret, cliprange, mpc_weight, mpc_suggested_ac], 
        losses + [_train]
    )
    
    # Setup counters first
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards
    # Use a simple global variable to track episodes across iterations
    global_episodes_count = 0
    
    def make_generator(current_iter=0):
        return traj_segment_generator(
            env, 
            pi, 
            timesteps_per_batch=timesteps_per_batch, 
            stochastic=True,
            mpc_weight_param=mpc_integration_weight_param,
            start_episode=global_episodes_count,  # Use the global count for episodes
            iteration=current_iter  # Pass current iteration as a parameter
        )
    
    seg_gen = make_generator(iters_so_far)
    
    checkpoint_dir = os.path.join(ROOT_PATH, "ckpts", saved_model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoints to {checkpoint_dir}")
    saver = tf.train.Saver(max_to_keep=10)
    
    print("Starting training...")
    
    # Setup metrics directories for logging and plotting
    metrics_dir = os.path.join(ROOT_PATH, "metrics", saved_model)
    plots_dir = os.path.join(metrics_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Metrics will be saved to {metrics_dir}")
    
    # Main training loop
    while timesteps_so_far < max_timesteps:
        # Set old parameter values to new parameters
        assign_old_eq_new()
        
        # Print iteration start message
        print(f"\n===== STARTING ITERATION {iters_so_far+1}/{int(max_timesteps/timesteps_per_batch)} =====")
        print(f"Total timesteps: {timesteps_so_far}/{max_timesteps} ({(timesteps_so_far/max_timesteps)*100:.1f}% complete)")
        
        # Create a new generator with the current iteration number
        if iters_so_far > 0:  # Only recreate after first iteration
            seg_gen = make_generator(iters_so_far)
        
        # Initialize newlosses to avoid UnboundLocalError
        newlosses = [0.0, 0.0, 0.0, 0.0, 0.0]  # Default values for policy, entropy, value, total, mpc losses
        
        try:
            # Get training batch
            seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)
            
            # Fix: Create consistent key names and provide defaults if keys are missing
            ob_vf = seg["ob_vf"]
            ac = seg["ac"]
            atarg = seg["adv"]
            tdlamret = seg["tdlamret"]
            
            # Fix: Use the correct key name "ep_rew" instead of "ep_rews"
            if "ep_rew" in seg:
                ep_rews = seg["ep_rew"]
            else:
                print("Warning: ep_rew key not found in segment, using empty list")
                ep_rews = []
                
            if "ep_lens" in seg:
                ep_lens = seg["ep_lens"]
            else:
                print("Warning: ep_lens key not found in segment, using empty list")
                ep_lens = []
            
            # Make sure to initialize mpc_acs and stability_metrics
            if "mpc_acs" in seg:
                mpc_acs = seg["mpc_acs"]
            else:
                print("Warning: mpc_acs key not found in segment, using zeros")
                mpc_acs = np.zeros((len(ob_vf), env.num_motor))
                
            if "stability_metrics" in seg:
                stability_metrics = seg["stability_metrics"]
            else:
                print("Warning: stability_metrics key not found in segment, using zeros")
                stability_metrics = np.zeros((len(ob_vf), 3))
            
            # Get policy observations
            if "ob_pol" in seg:
                ob_pol = seg["ob_pol"]
            else:
                print("Warning: ob_pol key not found in segment, using default shape")
                ob_pol = np.zeros((len(ob_vf), 10, 20))
                
            if "ob_pol_cnn" in seg:
                ob_pol_cnn = seg["ob_pol_cnn"]
            else:
                print("Warning: ob_pol_cnn key not found in segment, using default shape")
                ob_pol_cnn = np.zeros((len(ob_vf), 37, 60, 1))
            
            # Update buffers with episode data
            if len(ep_lens) > 0:
                lenbuffer.extend(ep_lens)
            if len(ep_rews) > 0:
                rewbuffer.extend(ep_rews)
                
            # Update episode counter only if we have episode data
            episodes_so_far += len(ep_lens)
            
            # Print episode statistics for monitoring
            if len(rewbuffer) > 0:
                print(f"Recent episodes: mean length={np.mean(lenbuffer):.2f}, mean reward={np.mean(rewbuffer):.2f}")
            
            # Compute current MPC integration weight based on stability metrics
            # Higher weight when robot is less stable
            avg_stability = np.mean(np.abs(stability_metrics)) if stability_metrics.size > 0 else 0.0
            current_mpc_weight = mpc_integration_weight_param * (1.0 + 5.0 * avg_stability)
            
            # Normalize the advantages
            atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-8)
            
            # Set update range
            optim_batchsize = min(optim_batchsize, ob_vf.shape[0])
            indexes = np.arange(len(ob_vf))
            
            # Learning rate decay
            cur_lrmult = 1.0 - timesteps_so_far / max_timesteps
            cliprangenow = clip_param * cur_lrmult
            
            # Do multiple epochs of optimization
            for _ in range(optim_epochs):
                np.random.shuffle(indexes)
                for start in range(0, len(indexes), optim_batchsize):
                    end = start + optim_batchsize
                    mbatch = indexes[start:end]
                    
                    try:
                        # Verify the batch shapes before passing to compute_losses
                        batch_ob_vf = ob_vf[mbatch]
                        
                        # Handle ob_pol based on its type and shape
                        if isinstance(ob_pol, np.ndarray):
                            if len(ob_pol) == len(ob_vf):
                                batch_ob_pol = ob_pol[mbatch]
                            else:
                                # Create properly sized array if dimensions don't match
                                expected_pol_shape = (len(mbatch), *ob_pol.shape[1:]) if ob_pol.ndim > 1 else (len(mbatch),)
                                batch_ob_pol = np.zeros(expected_pol_shape)
                                print(f"Warning: ob_pol shape mismatch, expected {expected_pol_shape}, got {ob_pol.shape}")
                        else:
                            # Use a default shape for ob_pol if it's not an array
                            batch_ob_pol = np.zeros((len(mbatch), 10, 20))
                            print(f"Warning: ob_pol is not an array, type is {type(ob_pol)}")
                        
                        # Handle ob_pol_cnn based on its type and shape
                        if isinstance(ob_pol_cnn, np.ndarray):
                            if len(ob_pol_cnn) == len(ob_vf):
                                batch_ob_pol_cnn = ob_pol_cnn[mbatch]
                            else:
                                # Create correctly shaped CNN input
                                batch_ob_pol_cnn = np.zeros((len(mbatch), 37, 60, 1))
                                print(f"Warning: ob_pol_cnn shape mismatch, expected {(len(mbatch), 37, 60, 1)}, got {ob_pol_cnn.shape}")
                        else:
                            # Use a default CNN shape
                            batch_ob_pol_cnn = np.zeros((len(mbatch), 37, 60, 1))
                            print(f"Warning: ob_pol_cnn is not an array, type is {type(ob_pol_cnn)}")
                            
                        batch_ac = ac[mbatch]
                        batch_atarg = atarg[mbatch]
                        batch_tdlamret = tdlamret[mbatch]
                        
                        # Handle MPC actions
                        if isinstance(mpc_acs, np.ndarray) and len(mpc_acs) == len(ob_vf):
                            batch_mpc_acs = mpc_acs[mbatch]
                        else:
                            batch_mpc_acs = np.zeros((len(mbatch), env.num_motor))
                            if isinstance(mpc_acs, np.ndarray):
                                print(f"Warning: mpc_acs shape mismatch, expected {(len(mbatch), env.num_motor)}, got {mpc_acs.shape}")
                            else:
                                print(f"Warning: mpc_acs is not an array, type is {type(mpc_acs)}")
                        
                        # Log first batch shapes in first iteration for debugging
                        if iters_so_far == 0 and start == 0:
                            print("First batch shapes:")
                            print(f"  ob_vf: {batch_ob_vf.shape}")
                            print(f"  ob_pol: {batch_ob_pol.shape if isinstance(batch_ob_pol, np.ndarray) else type(batch_ob_pol)}")
                            print(f"  ob_pol_cnn: {batch_ob_pol_cnn.shape if isinstance(batch_ob_pol_cnn, np.ndarray) else type(batch_ob_pol_cnn)}")
                            print(f"  ac: {batch_ac.shape}")
                            print(f"  atarg: {batch_atarg.shape}")
                            print(f"  tdlamret: {batch_tdlamret.shape}")
                            print(f"  mpc_acs: {batch_mpc_acs.shape}")
                        
                        # Check if any batch size is 0 (which could cause reshape errors)
                        if any(x.size == 0 for x in [batch_ob_vf, batch_ac, batch_atarg, batch_tdlamret, batch_mpc_acs] 
                              if isinstance(x, np.ndarray)):
                            print("Warning: Empty batch detected, skipping update")
                            continue
                            
                        # Check if batch_ob_pol and batch_ob_pol_cnn have the right shapes
                        if isinstance(batch_ob_pol, np.ndarray) and batch_ob_pol.size == 0:
                            print(f"Warning: Empty ob_pol batch, using default shape")
                            batch_ob_pol = np.zeros((len(mbatch), 10, 20))
                            
                        if isinstance(batch_ob_pol_cnn, np.ndarray) and batch_ob_pol_cnn.size == 0:
                            print(f"Warning: Empty ob_pol_cnn batch, using default shape")
                            batch_ob_pol_cnn = np.zeros((len(mbatch), 37, 60, 1))
                        
                        # Do gradient update with proper error handling
                        try:
                            *newlosses, _ = compute_losses(
                                batch_ob_vf, 
                                batch_ob_pol, 
                                batch_ob_pol_cnn, 
                                batch_ac, 
                                batch_atarg, 
                                batch_tdlamret, 
                                cliprangenow,
                                current_mpc_weight,
                                batch_mpc_acs
                            )
                        except ValueError as ve:
                            if "cannot reshape array of size 2 into shape" in str(ve):
                                print("Known shape mismatch error detected. Printing detailed shape info:")
                                print(f"  ob_vf: {batch_ob_vf.shape}")
                                print(f"  ob_pol: {batch_ob_pol.shape if isinstance(batch_ob_pol, np.ndarray) else type(batch_ob_pol)}")
                                print(f"  ob_pol_cnn: {batch_ob_pol_cnn.shape if isinstance(batch_ob_pol_cnn, np.ndarray) else type(batch_ob_pol_cnn)}")
                                print(f"  ac: {batch_ac.shape}")
                                print(f"  atarg: {batch_atarg.shape}")
                                print(f"  tdlamret: {batch_tdlamret.shape}")
                                print(f"  mpc_acs: {batch_mpc_acs.shape}")
                                
                                # Additional debug info for the reshape error
                                print("\nSpecific error:", ve)
                                print("\nSome array content samples:")
                                print(f"  batch_ob_pol sample (first 5 elements):", 
                                     batch_ob_pol.flatten()[:5] if isinstance(batch_ob_pol, np.ndarray) and batch_ob_pol.size > 0 else "Empty")
                                print(f"  batch_ob_pol_cnn shape:", batch_ob_pol_cnn.shape if isinstance(batch_ob_pol_cnn, np.ndarray) else "Not an array")
                                
                                # Skip this batch but continue training
                                continue
                            else:
                                # Re-raise unexpected errors
                                raise
                    except Exception as e:
                        print(f"Warning: Skipping batch update due to error: {str(e)}")
                        # Continue with next batch
                        continue
        except Exception as e:
            print(f"Error during iteration: {e}")
            print("Resetting environment and continuing to next iteration...")
            # Reset any internal variables to a safe state
            ob_vf = None
            ob_pol = None
            ob_pol_cnn = None
            ep_rews = []
            ep_lens = []
            mpc_acs = None
            stability_metrics = np.zeros((1, 3))  # Ensure this is defined with a safe default
            
            # Reset environment to recover
            try:
                env.reset()
            except Exception as reset_error:
                print(f"Warning: Error during environment reset: {reset_error}")
                
            # Just update counters and continue to next iteration
            iters_so_far += 1
            continue
        
        # Update counters
        lenbuffer.extend(ep_lens)
        rewbuffer.extend(ep_rews)
        episodes_so_far += len(ep_lens)
        timesteps_so_far += sum(ep_lens)
        iters_so_far += 1
        
        # Update global episode counter for the next iteration
        global_episodes_count = episodes_so_far
        
        # Create a visual progress bar
        progress_percent = timesteps_so_far / max_timesteps
        bar_length = 30
        filled_length = int(bar_length * progress_percent)
        progress_bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Calculate estimated time remaining
        elapsed = time.time() - tstart
        if iters_so_far > 0:
            fps = timesteps_so_far / elapsed
            steps_left = max_timesteps - timesteps_so_far
            est_seconds_left = steps_left / fps if fps > 0 else 0
            
            # Convert to hours, minutes, seconds
            hours, remainder = divmod(est_seconds_left, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            print(f"\n===== COMPLETED ITERATION {iters_so_far}/{int(max_timesteps/timesteps_per_batch)} =====")
            print(f"[{progress_bar}] {progress_percent*100:.1f}%")
            print(f"Episodes: {episodes_so_far}, Timesteps: {timesteps_so_far}/{max_timesteps}")
            print(f"Training speed: {int(fps)} steps/sec | Est. time remaining: {time_str}")
        else:
            print(f"\n===== COMPLETED ITERATION {iters_so_far}/{int(max_timesteps/timesteps_per_batch)} =====")
            print(f"[{progress_bar}] {progress_percent*100:.1f}%")
            print(f"Episodes: {episodes_so_far}, Timesteps: {timesteps_so_far}/{max_timesteps}")
        
        # Log statistics and save data/plots every 10 iterations
        if iters_so_far % 10 == 0:
            elapsed = time.time() - tstart
            fps = int(timesteps_so_far / elapsed)
            
            # Log basic stats to terminal
            print(f"------------ Iteration {iters_so_far}/{int(max_timesteps/timesteps_per_batch)} ------------")
            print(f"Total timesteps: {timesteps_so_far}")
            print(f"Total episodes: {episodes_so_far}")
            print(f"Mean reward: {np.mean(rewbuffer):.3f}")
            print(f"Mean episode length: {np.mean(lenbuffer):.3f}")
            print(f"Training FPS: {fps}")
            
            # Display only key metrics, not all losses - with a safety check
            if len(newlosses) >= 3:
                print(f"Policy loss: {newlosses[0]:.4f}, Value loss: {newlosses[2]:.4f}")
            else:
                print("Policy loss and Value loss not available for this iteration")
            print(f"MPC weight: {current_mpc_weight:.3f}")
            print("----------------------------------------")
            sys.stdout.flush()
            
            # Save training metrics and generate plots
            try:
                # Save metrics to CSV
                metrics_data = {
                    'iteration': iters_so_far,
                    'timesteps': timesteps_so_far,
                    'episodes': episodes_so_far,
                    'mean_reward': np.mean(rewbuffer) if len(rewbuffer) > 0 else 0,
                    'std_reward': np.std(rewbuffer) if len(rewbuffer) > 0 else 0,
                    'min_reward': np.min(rewbuffer) if len(rewbuffer) > 0 else 0,
                    'max_reward': np.max(rewbuffer) if len(rewbuffer) > 0 else 0,
                    'mean_ep_length': np.mean(lenbuffer) if len(lenbuffer) > 0 else 0,
                    'policy_loss': newlosses[0] if len(newlosses) > 0 else 0,
                    'value_loss': newlosses[2] if len(newlosses) > 2 else 0,
                    'fps': fps,
                    'current_mpc_weight': current_mpc_weight,
                    'elapsed_time': elapsed
                }
                
                # Save metrics to CSV
                metrics_file = os.path.join(metrics_dir, "training_metrics.csv")
                # Check if file exists to append or create new
                if os.path.exists(metrics_file):
                    metrics_df = pd.read_csv(metrics_file)
                    # Append new row, first checking if this iteration already exists
                    if not (metrics_df['iteration'] == iters_so_far).any():
                        new_row = pd.DataFrame([metrics_data])
                        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
                        metrics_df.to_csv(metrics_file, index=False)
                else:
                    # Create new file
                    pd.DataFrame([metrics_data]).to_csv(metrics_file, index=False)
                
                # Generate and save plots
                plot_iteration = os.path.join(plots_dir, f"iter_{iters_so_far:06d}")
                os.makedirs(plot_iteration, exist_ok=True)
                
                # Plot reward history
                if os.path.exists(metrics_file):
                    try:
                        df = pd.read_csv(metrics_file)
                        
                        # Plot training progress
                        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                        fig.suptitle(f'Training Progress - Iteration {iters_so_far}', fontsize=16)
                        
                        # Plot rewards
                        ax = axes[0, 0]
                        ax.plot(df['iteration'], df['mean_reward'], 'b-', label='Mean Reward')
                        if len(df) > 1:
                            ax.fill_between(
                                df['iteration'],
                                df['min_reward'],
                                df['max_reward'],
                                alpha=0.2, color='blue'
                            )
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel('Reward')
                        ax.set_title('Reward over Training')
                        ax.legend()
                        ax.grid(True)
                        
                        # Plot episode lengths
                        ax = axes[0, 1]
                        ax.plot(df['iteration'], df['mean_ep_length'], 'g-')
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel('Mean Episode Length')
                        ax.set_title('Episode Length over Training')
                        ax.grid(True)
                        
                        # Plot losses
                        ax = axes[1, 0]
                        ax.plot(df['iteration'], df['policy_loss'], 'r-', label='Policy Loss')
                        ax.plot(df['iteration'], df['value_loss'], 'm-', label='Value Loss')
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel('Loss')
                        ax.set_title('Training Losses')
                        ax.legend()
                        ax.grid(True)
                        
                        # Plot MPC weight
                        ax = axes[1, 1]
                        ax.plot(df['iteration'], df['current_mpc_weight'], 'c-')
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel('MPC Weight')
                        ax.set_title('MPC Integration Weight')
                        ax.grid(True)
                        
                        plt.tight_layout(rect=[0, 0, 1, 0.96])
                        plt.savefig(os.path.join(plot_iteration, 'training_progress.png'), dpi=150)
                        plt.close(fig)
                        
                        print(f"Saved training plots to {plot_iteration}")
                    except Exception as plot_error:
                        print(f"Warning: Error generating plots: {plot_error}")
            except Exception as metrics_error:
                print(f"Warning: Error saving metrics: {metrics_error}")
        
        # Save model checkpoint (at original frequency - every 100 iterations)
        if (timesteps_so_far > 0 and timesteps_so_far % save_freq == 0) or iters_so_far == 0:
            save_path = os.path.join(checkpoint_dir, f"model_{timesteps_so_far}")
            print(f"Saving model to {save_path}")
            saver.save(sess, save_path)
    
    # Final save
    save_path = os.path.join(checkpoint_dir, f"model_final_{timesteps_so_far}")
    saver.save(sess, save_path)
    env.close()
    
    return pi


def train_standard(max_iters, with_gpu=False, callback=None):
    """Standard training method without the enhanced MPC integration"""
    # training define
    if not with_gpu:
        config = tf.ConfigProto(device_count={"GPU": 0})
        U.make_session(config=config).__enter__()
        print("**************Using CPU**************")
    else:
        U.make_session().__enter__()
        print("**************Using GPU**************")

    def policy_fn(name, ob_space_vf, ob_space_pol, ob_space_pol_cnn, ac_space):
        return MLPCNNPolicy(
            name=name,
            ob_space_vf=ob_space_vf,
            ob_space_pol=ob_space_pol,
            ob_space_pol_cnn=ob_space_pol_cnn,
            ac_space=ac_space,
            hid_size=512,
            num_hid_layers=2,
        )

    # Create a copy of the config and update MPC setting
    mpc_config = config_mpc_train.copy()
    mpc_config["use_mpc"] = use_mpc
    
    # Ensure all required keys are present
    required_keys = [
        "fixed_gait", "add_standing", "add_rotation", "fixed_gait_cmd",
        "max_timesteps", "is_visual", "cam_track_robot", "step_zerotorque",
        "minimal_rand", "is_noisy", "add_perturbation", "use_action_filter"
    ]
    
    # Set default values for any missing keys
    defaults = {
        "fixed_gait": True,
        "add_standing": True,
        "add_rotation": True,
        "fixed_gait_cmd": [0.0, 0.0, 0.98, 0.0],  # vx, vy, height, rotation
        "max_timesteps": 8000,
        "is_visual": False,
        "cam_track_robot": False,
        "step_zerotorque": False,
        "minimal_rand": False,
        "is_noisy": True,
        "add_perturbation": True,
        "use_action_filter": True
    }
    
    for key in required_keys:
        if key not in mpc_config:
            mpc_config[key] = defaults[key]
            print(f"Warning: Added missing key '{key}' with default value: {defaults[key]}")
    
    env = CassieEnv(config=mpc_config)

    from ppo import ppo_sgd_cnn as ppo_sgd
    '''
    if MLP (short history only) 
    from ppo import ppo_sgd_mlp as ppo_sgd. 
    CassieEnv needs to change to short history setup (not implemented but easy to change)
    '''

    pi = ppo_sgd.learn(
        env,
        policy_fn,
        max_iters=max_iters,
        timesteps_per_actorbatch=4096,
        clip_param=0.2,
        entcoeff=0,
        optim_epochs=2,
        optim_stepsize=1e-4,
        optim_batchsize=512,
        gamma=0.98,
        lam=0.95,
        callback=callback,
        schedule="constant",
        continue_from=restore_model_from_file,
    )
    return pi


def training_callback(locals_, globals_):
    saver_ = locals_["saver"]
    sess_ = U.get_session()
    timesteps_so_far_ = locals_["timesteps_so_far"]
    iters_so_far_ = locals_["iters_so_far"]
    model_dir = model_folder + saved_model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if MPI.COMM_WORLD.Get_rank() == 0 and iters_so_far_ % save_interval == 0:
        saver_.save(sess_, model_dir + "/model", global_step=timesteps_so_far_)
    return True


if __name__ == "__main__":
    logger.configure()
    
    # Set random seed
    set_global_seeds(rnd_seed)
    
    # Use enhanced MPC integration if MPC is enabled
    if use_mpc:
        print("Using enhanced MPC integration training...")
        learn_with_mpc(
            max_timesteps=max_iters * 4096,  # Convert iterations to timesteps
            eval_freq=save_interval * 4096,
            save_freq=save_interval * 4096,
            seed=rnd_seed,
            gamma=gamma,
            lam=lam,
            lr=lr,
            optim_stepsize=lr,
            mpc_integration_weight_param=mpc_weight,
            visualize=visualize
        )
    else:
        print("Using standard training without MPC integration...")
        train_standard(max_iters=max_iters, callback=training_callback) 