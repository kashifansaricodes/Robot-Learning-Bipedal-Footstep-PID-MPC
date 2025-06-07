#!/usr/bin/env python
import os
import sys
import time
from collections import deque
import numpy as np
import tensorflow as tf
from baselines.common import explained_variance
from baselines.common import tf_util as U
import ppo.policies as policies

from rlenv.cassie_env import CassieEnv
from configs.defaults import ROOT_PATH
from configs.env_config import config_mpc_train

# set numpy print options
np.set_printoptions(precision=3, suppress=True)

def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0
    
    myseed = i  + 1000 * rank if i is not None else None
    tf.set_random_seed(myseed)
    np.random.seed(myseed)

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
    mpc_integration_weight_param=0.1  # Parameter to control how much to blend MPC suggestions
):
    """
    Train an agent using PPO with MPC integration on the Cassie robot
    """
    # 1. Create Environment
    # Configure environment to use MPC
    config = config_mpc_train.copy()
    config["use_mpc"] = True
    config["mpc_horizon"] = 10
    config["mpc_dt"] = 0.01
    config["max_timesteps"] = 5000
    config["is_visual"] = False
    config["cam_track_robot"] = False
    
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
    
    cliprange = tf.placeholder(dtype=tf.float32, shape=[])
    mpc_weight = tf.placeholder(dtype=tf.float32, shape=[])  # Control MPC integration weight
    
    vf_coef = vf_coeff
    entcoeff = entcoeff
    
    # Create PPO loss function
    ob_vf = U.get_placeholder_cached(name="ob_vf")
    ac = pi.pdtype.sample_placeholder([None])
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return
    mpc_suggested_ac = tf.placeholder(dtype=tf.float32, shape=[None, env.num_motor])  # MPC suggested action
    
    # Extract MPC suggestions from observations (last 13 elements: 10 motor commands + 3 stability metrics)
    # Assuming observations include MPC suggestions at the end
    mpc_suggestion_idx = -13  # Position where MPC suggestions start in observation
    
    # Get PPO action distribution and values
    ac_pred, vpred = pi._act(stochastic=True, ob_vf=ob_vf)
    
    # Create loss function that integrates MPC suggestions
    old_vpred = tf.placeholder(dtype=tf.float32, shape=[None])
    old_ac_pred = tf.placeholder(dtype=tf.float32, shape=[None, ac_space.shape[0]])
    
    # Regular PPO loss calculation
    ratio = tf.exp(pi.pd.logp(ac) - pi.old_pd.logp(ac))
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
    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in zip(pi.old_params, pi.get_params())])
    
    # Function to compute losses and do optimization
    compute_losses = U.function(
        [ob_vf, ac, atarg, ret, cliprange, mpc_weight, mpc_suggested_ac], 
        losses + [_train]
    )
    
    # Setup training loop
    seg_gen = traj_segment_generator(
        env, 
        pi, 
        timesteps_per_batch=2048, 
        stochastic=True,
        mpc_weight_param=mpc_integration_weight_param
    )
    
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards
    
    checkpoint_dir = os.path.join(ROOT_PATH, "ckpts", "mpc_integrated")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoints to {checkpoint_dir}")
    saver = tf.train.Saver(max_to_keep=10)
    
    print("Starting training...")
    
    # Main training loop
    while timesteps_so_far < max_timesteps:
        # Set old parameter values to new parameters
        assign_old_eq_new()
        
        # Get training batch
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        
        # Get all input for training
        ob_vf, ac, atarg, tdlamret = seg["ob_vf"], seg["ac"], seg["adv"], seg["tdlamret"]
        mpc_acs = seg["mpc_acs"]  # MPC suggested actions
        stability_metrics = seg["stability_metrics"]  # Stability metrics from MPC
        
        # Compute current MPC integration weight based on stability metrics
        # Higher weight when robot is less stable
        avg_stability = np.mean(np.abs(stability_metrics))
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
                
                # Do gradient update
                *newlosses, _ = compute_losses(
                    ob_vf[mbatch], 
                    ac[mbatch], 
                    atarg[mbatch], 
                    tdlamret[mbatch], 
                    cliprangenow,
                    current_mpc_weight,
                    mpc_acs[mbatch]
                )
        
        # Update counters
        lenbuffer.extend(seg["ep_lens"])
        rewbuffer.extend(seg["ep_rews"])
        episodes_so_far += len(seg["ep_lens"])
        timesteps_so_far += sum(seg["ep_lens"])
        iters_so_far += 1
        
        # Log and save
        if iters_so_far % 10 == 0:
            elapsed = time.time() - tstart
            fps = int(timesteps_so_far / elapsed)
            
            print(f"------------ Iteration {iters_so_far} ------------")
            print(f"timesteps: {timesteps_so_far}")
            print(f"episodes: {episodes_so_far}")
            print(f"mean reward: {np.mean(rewbuffer):.3f}")
            print(f"mean episode length: {np.mean(lenbuffer):.3f}")
            print(f"fps: {fps}")
            
            for name, loss in zip(loss_names, newlosses):
                print(f"{name}: {loss:.6f}")
            
            print(f"MPC integration weight: {current_mpc_weight:.6f}")
            print("----------------------------------------")
            sys.stdout.flush()
        
        # Save model
        if (timesteps_so_far > 0 and timesteps_so_far % save_freq == 0) or iters_so_far == 0:
            save_path = os.path.join(checkpoint_dir, f"model_{timesteps_so_far}")
            print(f"Saving model to {save_path}")
            saver.save(sess, save_path)
    
    # Final save
    save_path = os.path.join(checkpoint_dir, f"model_final_{timesteps_so_far}")
    saver.save(sess, save_path)
    env.close()
    
    return pi

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

def traj_segment_generator(env, policy, timesteps_per_batch, stochastic, mpc_weight_param=0.1):
    """
    Generate trajectories for training
    """
    # Initialize buffer
    t = 0
    ac = np.zeros(env.action_space.shape[0], dtype=np.float32)
    
    # Setup buffers for storing trajectories
    ob_vf, _ = env.reset()
    ob_vf_dim = ob_vf.shape[0]
    
    # Indices for MPC suggestions and stability metrics in observations
    mpc_start_idx = -13  # Position where MPC suggestions start
    stability_metrics_idx = -3  # Position where stability metrics start
    
    # Get dimensions for ac_pred and vpred
    stochastic_ac, vpred_val = policy.act(stochastic=stochastic, ob_vf=ob_vf)
    
    # Buffers
    new = True
    rew = 0.0
    ob_vfs = np.array([ob_vf for _ in range(timesteps_per_batch)])
    acs = np.array([ac for _ in range(timesteps_per_batch)])
    vpreds = np.zeros(timesteps_per_batch, 'float32')
    news = np.zeros(timesteps_per_batch, 'int32')
    rews = np.zeros(timesteps_per_batch, 'float32')
    ep_rews = []  # episode rewards
    ep_lens = []  # episode lengths
    
    # Add buffers for MPC suggestions and stability metrics
    mpc_acs = np.zeros((timesteps_per_batch, env.num_motor), dtype=np.float32)
    stability_metrics = np.zeros((timesteps_per_batch, 3), dtype=np.float32)  # roll, pitch, z_vel
    
    # Loop through and collect trajectories
    while True:
        # Save observations
        ob_vfs[t] = ob_vf
        vpreds[t] = vpred_val
        news[t] = new
        
        # Extract MPC suggestions and stability metrics from observations
        mpc_acs[t] = ob_vf[mpc_start_idx:mpc_start_idx+env.num_motor]
        stability_metrics[t] = ob_vf[stability_metrics_idx:stability_metrics_idx+3]
        
        # Take action
        ac, vpred_val = policy.act(stochastic=stochastic, ob_vf=ob_vf)
        
        # Blend MPC suggestion with policy action when needed based on stability
        roll_pitch_tilt = np.abs(stability_metrics[t][:2])  # roll and pitch tilt
        z_vel = stability_metrics[t][2]  # z velocity
        
        # Determine if we need MPC help
        needs_stabilization = False
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
        ob_vf, _, rew, new, info = env.step(ac)
        rews[t] = rew
        
        # Increment counter
        t += 1
        
        # Check if we have enough transitions
        if t >= timesteps_per_batch:
            # Add final return value to the last state
            _, vpred_val = policy.act(stochastic=stochastic, ob_vf=ob_vf)
            
            # Cut buffers to actual size
            ob_vfs = ob_vfs[:t]
            acs = acs[:t]
            vpreds = vpreds[:t]
            news = news[:t]
            rews = rews[:t]
            mpc_acs = mpc_acs[:t]
            stability_metrics = stability_metrics[:t]
            
            # Return trajectory data
            yield {
                "ob_vf": ob_vfs,
                "ac": acs,
                "vpred": vpreds,
                "nextvpred": vpred_val,
                "new": news,
                "rew": rews,
                "ep_rew": ep_rews,
                "ep_lens": ep_lens,
                "mpc_acs": mpc_acs,
                "stability_metrics": stability_metrics
            }
            
            # Reset buffers
            t = 0
            ob_vfs = np.array([ob_vf for _ in range(timesteps_per_batch)])
            acs = np.array([ac for _ in range(timesteps_per_batch)])
            vpreds = np.zeros(timesteps_per_batch, 'float32')
            news = np.zeros(timesteps_per_batch, 'int32')
            rews = np.zeros(timesteps_per_batch, 'float32')
            mpc_acs = np.zeros((timesteps_per_batch, env.num_motor), dtype=np.float32)
            stability_metrics = np.zeros((timesteps_per_batch, 3), dtype=np.float32)
            ep_rews = []
            ep_lens = []
        
        # Handle episode completion
        if new:
            ep_rews.append(rew)
            ep_lens.append(t)
            rew = 0
            t = 0

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_global_seeds(42)
    
    # Train the model
    learn_with_mpc(
        max_timesteps=2e6,
        eval_freq=5e3,
        save_freq=5e3,
        mpc_integration_weight_param=0.1
    ) 