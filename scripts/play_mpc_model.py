#!/usr/bin/env python
import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from baselines.common import tf_util as U

from rlenv.cassie_env import CassieEnv
import ppo.policies as policies
from configs.env_config import config_play
from configs.defaults import ROOT_PATH

# to ignore specific deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

model_folder = ROOT_PATH + "/ckpts/"

def get_args():
    parser = argparse.ArgumentParser(description="MPC Model Testing")

    parser.add_argument(
        "--model_name", type=str, default="new_training_rnds1", help="model name to test (default: new_training_rnds1)"
    )
    parser.add_argument(
        "--episode_len", type=int, default=5000, help="episode length to test (default: 5000)"
    )
    parser.add_argument(
        "--use_mpc", action="store_true", default=False, help="use MPC for additional stability (default: False)"
    )
    parser.add_argument(
        "--render_only", action="store_true", default=False, help="just render without running policy (default: False)"
    )
    parser.add_argument(
        "--record_video", action="store_true", default=False, help="record video (default: False)"
    )
    
    args = parser.parse_args()
    return args

def main():
    """
    Load and test a trained model from checkpoints.
    """
    args = get_args()
    
    # Update configuration for testing
    config = config_play.copy()
    
    # Enable MPC if requested
    if args.use_mpc:
        config["use_mpc"] = True
        config["mpc_horizon"] = 10
        config["mpc_dt"] = 0.01
        print("Using MPC for additional stability")
    else:
        config["use_mpc"] = False
        print("Running without MPC")
    
    # Create environment
    env = CassieEnv(config=config)
    
    # Get checkpoint path
    model_dir = os.path.join(model_folder, args.model_name)
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    
    if latest_checkpoint is None:
        print(f"No checkpoint found in {model_dir}")
        available_models = os.listdir(model_folder)
        print(f"Available models: {available_models}")
        return
    
    print(f"Loading model from: {latest_checkpoint}")
    model_path = latest_checkpoint
    
    # Configure session to use CPU
    config = tf.ConfigProto(device_count={"GPU": 0})
    session = U.make_session(config=config)

    # Create policy
    ob_space_pol = env.observation_space_pol
    ac_space = env.action_space
    ob_space_vf = env.observation_space_vf
    ob_space_pol_cnn = env.observation_space_pol_cnn
    
    pi = policies.MLPCNNPolicy(
        name="pi",
        ob_space_vf=ob_space_vf,
        ob_space_pol=ob_space_pol,
        ob_space_pol_cnn=ob_space_pol_cnn,
        ac_space=ac_space,
        hid_size=512,
        num_hid_layers=2,
    )

    # Load trained model
    U.initialize()
    U.load_state(model_path)
    
    # Test the model
    rewards = []
    episode_reward = 0
    
    # Attempt to initialize OpenGL with safer settings
    try:
        # Force software rendering
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        # Set OpenGL version
        os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
        
        # Check if we're on Linux and try to use EGL backend
        if sys.platform.startswith('linux'):
            os.environ['MUJOCO_GL'] = 'egl'
    except Exception as e:
        print(f"Warning: Failed to set OpenGL context variables: {e}")
    
    # Start visualization
    try:
        draw_state = env.render()
    except Exception as e:
        print(f"Error initializing rendering: {e}")
        print("Try running with different GL settings or in headless mode.")
        return
    
    # If just rendering without policy
    if args.render_only:
        print("Rendering only mode - no policy execution")
        while draw_state:
            draw_state = env.render()
            time.sleep(0.01)
        return
    
    # Main test loop
    print(f"Starting test of model {args.model_name} for {args.episode_len} steps")
    total_steps = 0
    
    while draw_state and total_steps < args.episode_len:
        # Reset environment
        obs_vf, obs_pol = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"Starting new episode")
        
        # Run episode
        while not done and steps < args.episode_len and draw_state:
            if not env.vis.ispaused():
                # Get action from policy
                ac = pi.act(stochastic=False, ob_vf=obs_vf, ob_pol=obs_pol)[0]
                
                # If using MPC, we can blend with MPC suggestions
                if args.use_mpc:
                    try:
                        # Get MPC action suggestion
                        current_state = env._get_current_state()
                        reference_trajectory = env._generate_mpc_reference_trajectory()
                        mpc_action = env.mpc_controller.compute_control(current_state, reference_trajectory)
                        
                        # Check stability metrics to determine blend weight
                        base_orientation = env.qpos[3:7]  # quaternion
                        z_velocity = env.qvel[2]
                        
                        # Convert quaternion to euler angles
                        from utility.utility import quat2euler
                        euler_angles = quat2euler(base_orientation)
                        roll_pitch = np.abs(euler_angles[:2])
                        
                        # Determine if robot needs stabilization
                        needs_stabilization = np.any(roll_pitch > 0.15) or z_velocity < -0.3
                        
                        # Blend policy and MPC actions if needed
                        if needs_stabilization:
                            # Calculate blend weight based on severity
                            severity = max(
                                np.max(roll_pitch) / 0.2,
                                abs(min(z_velocity, 0)) / 0.4
                            )
                            # Cap at 1.0
                            severity = min(severity, 1.0)
                            
                            # Adaptive blend - more MPC influence when unstable
                            mpc_weight = 0.3 * severity
                            ac = (1 - mpc_weight) * ac + mpc_weight * mpc_action
                            if severity > 0.5:
                                print(f"MPC stabilization applied with weight {mpc_weight:.2f}")
                    except Exception as e:
                        print(f"MPC stabilization error: {e}")
                
                # Step environment
                obs_vf, obs_pol, reward, done, info = env.step(ac)
                episode_reward += reward
                steps += 1
                total_steps += 1
                
                # Render
                draw_state = env.render()
                
                # Print progress
                if steps % 100 == 0:
                    print(f"Step {steps}, Episode Reward: {episode_reward:.2f}")
                
                # Check if done
                if done:
                    print(f"Episode finished after {steps} steps with reward {episode_reward:.2f}")
                    rewards.append(episode_reward)
                    break
            else:
                # If paused, just render
                while env.vis.ispaused() and draw_state:
                    draw_state = env.render()
                    time.sleep(0.01)
    
    # Print final stats
    if rewards:
        print(f"Test completed. Average reward: {np.mean(rewards):.2f}")
        print(f"Total episodes: {len(rewards)}")
        print(f"Best episode: {np.max(rewards):.2f}")
        print(f"Worst episode: {np.min(rewards):.2f}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main() 