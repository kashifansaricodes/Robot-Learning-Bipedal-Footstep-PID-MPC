# Footstep Following Branch - Pure Reinforcement Learning

> **⚠️ Prerequisites Required**: Complete the full setup from the [**Main Branch README**](https://github.com/kashifansaricodes/Robot-Learning-Bipedal-Footstep-PID-MPC) before proceeding. This includes MuJoCo installation, Conda environment setup, and base dependencies.

This branch implements **pure reinforcement learning** for bipedal locomotion, focusing on end-to-end learning without explicit model-based control. The approach uses PPO (Proximal Policy Optimization) to learn walking policies directly from simulation experience.

## 🤖 Branch-Specific Technical Details

### Algorithm: Proximal Policy Optimization (PPO)
- **Policy Network**: Neural network mapping observations to continuous actions
- **Value Function**: Separate network for state value estimation  
- **Action Space**: Continuous joint torques/positions for all Cassie actuators
- **Observation Space**: Joint positions, velocities, IMU data, contact forces
- **Reward Design**: Encourages forward locomotion, stability, and energy efficiency

### Performance Targets
- **Walking Speed**: 1.0 m/s sustained forward velocity
- **Control Frequency**: 33 Hz (timestep = 0.03s)
- **Training Time**: 2M timesteps (~8-12 hours on 8-core CPU)
- **Episode Length**: 1000+ steps without falling

## 🔧 Branch-Specific Installation

> **📋 Prerequisite Check**: Ensure you've completed the setup from the [Main Branch README](https://github.com/kashifansaricodes/Robot-Learning-Bipedal-Footstep-PID-MPC) including MuJoCo 2.1.0 and base Conda environment.

After completing main branch setup:

```bash
# Switch to footstep_following branch
git checkout footstep_following

# Install OpenAI Baselines for PPO implementation
cd external/baselines
pip install -e .
cd ../..

# Install project in development mode
pip install -e .

# Verify installation
python -c "import mujoco_py; print('Setup verified!')"
```

## 🚀 Training & Testing

### Training a Policy

Navigate to the execution directory and start training:
```bash
cd cassieWalking/exe
./train.sh
```

**Training Script Details:**
The training uses parallel processing with MPI:
```bash
#!/bin/bash
mpirun -np 8 python ../src/train.py \
    --env_name=Cassie-v0 \
    --algorithm=ppo2 \
    --num_timesteps=2000000 \
    --save_interval=100
```

**Key Parameters to Customize:**
- `-np 8`: Number of parallel workers (adjust to your CPU cores)
- `--num_timesteps=2000000`: Total training steps (2M default)
- `--save_interval=100`: How often to save model checkpoints

### Testing Trained Models

Test a trained policy:
```bash
./test.sh
```

**Test Script Configuration:**
```bash
#!/bin/bash
python ../src/test.py \
    --test_model=ckpts/model_1000.pkl \
    --render=True \
    --num_episodes=10
```

### Training Phases & Progress

Expected training progression:
1. **Exploration Phase** (0-500K): Learning balance and basic coordination
2. **Locomotion Phase** (500K-1M): Forward walking patterns emerge  
3. **Optimization Phase** (1M-2M): Speed and efficiency improvements

**Monitor Progress:**
- **Console**: Real-time reward and performance metrics
- **Checkpoints**: Models saved in `ckpts/` directory
- **Logs**: Detailed training statistics

## 🎛️ Environment Configuration

### Observation Space (37-dimensional)
```python
observation = [
    joint_positions,      # (10D) Motor and joint angles
    joint_velocities,     # (10D) Angular velocities  
    imu_data,            # (7D) Orientation quaternion, angular velocity
    contact_forces,      # (4D) Left/right foot contact information
    pelvis_state,        # (6D) Position and velocity
]
```

### Action Space (10-dimensional)
```python
action = [
    motor_commands,      # (10D) Target positions/torques for actuated joints
]
# Range: [-1, 1] normalized and scaled to motor limits
# Frequency: 33 Hz control updates
```

### Reward Function Design
```python
def compute_reward(state, action, next_state):
    reward = (
        w1 * forward_velocity_reward +     # Encourages forward locomotion
        w2 * alive_bonus +                 # Reward for staying upright  
        w3 * energy_penalty +              # Minimizes actuator effort
        w4 * stability_reward              # Promotes balanced walking
    )
    return reward
```

## 📊 Performance Optimization

### Training Speed Tips
- **CPU Cores**: Match `-np` parameter to available cores (check with `nproc`)
- **Memory Usage**: Monitor with `htop` during training
- **Checkpointing**: Save frequently to avoid losing progress
- **Batch Size**: Larger batches can improve convergence

### Policy Performance Tips
- **Observation Normalization**: Critical for stable training convergence
- **Reward Weight Tuning**: Adjust `w1`, `w2`, `w3`, `w4` for desired walking behavior
- **Curriculum Learning**: Start with easier tasks, gradually increase difficulty
- **Network Architecture**: Balance model capacity with training speed

### Common Training Issues
> **📖 For installation issues**, refer to the troubleshooting section in the [Main Branch README](https://github.com/kashifansaricodes/Robot-Learning-Bipedal-Footstep-PID-MPC).

**Training-Specific Problems:**
```bash
# Slow training on multi-core systems
# Reduce workers if CPU usage is too high
mpirun -np 4 python train.py  # Instead of -np 8

# Memory issues during training
# Monitor memory usage and reduce batch size if needed

# Poor convergence
# Try different reward weights or learning rates
```

## 🔬 Research Applications

### Academic Use Cases
- **Bipedal Locomotion Research**: Baseline for comparing novel RL algorithms
- **Sim-to-Real Transfer**: Foundation for real robot deployment studies
- **Control Theory Comparison**: Benchmark against model-based approaches
- **Machine Learning Education**: Hands-on RL implementation example

### Experimental Extensions
- **Terrain Adaptation**: Train on varied ground surfaces and obstacles
- **Multi-Task Learning**: Incorporate different gaits (walking, running, jumping)  
- **Robustness Testing**: Systematic perturbation and noise studies
- **Ablation Studies**: Component-wise performance analysis

### Implementation Ideas
- **Custom Reward Functions**: Design rewards for specific behaviors
- **Observation Space Modifications**: Add/remove sensors or state information
- **Network Architecture Experiments**: Try different neural network designs
- **Transfer Learning**: Adapt policies to different robot morphologies

## 📁 Project Structure Overview
```
cassieWalking/
├── exe/
│   ├── train.sh          # Training script
│   ├── test.sh           # Testing script
│   └── ckpts/            # Model checkpoints directory
├── src/
│   ├── train.py          # PPO training implementation
│   ├── test.py           # Policy testing and evaluation
│   ├── cassie_env.py     # Cassie environment wrapper
│   └── policy_net.py     # Neural network architectures
├── external/
│   └── baselines/        # OpenAI Baselines integration
└── config/
    └── env_configs.py    # Environment parameter configurations
```

### Testing Your Changes
```bash
# Run basic functionality tests
python -c "import cassie_env; print('Environment loads successfully')"

# Quick training test (short run)
cd cassieWalking/exe
mpirun -np 2 python ../src/train.py --num_timesteps=1000

# Test saved models
python ../src/test.py --test_model=ckpts/latest.pkl --num_episodes=1
```

---

## 👥 Branch Contributor
- **Sounderya Varagur Venugopal** - Collaborating Researcher (UID: 121272423)

## 📚 Key References for This Branch
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Feedback Control For Cassie With Deep Reinforcement Learning](https://arxiv.org/abs/1803.05580)
- [Learning Locomotion Skills for Cassie: Iterative Design and Sim-to-Real](https://arxiv.org/abs/1909.05944)

---

## 🎯 Quick Start Checklist

- [ ] ✅ Completed [Main Branch README](https://github.com/kashifansaricodes/Robot-Learning-Bipedal-Footstep-PID-MPC) setup
- [ ] 🔄 Switched to `footstep_following` branch
- [ ] 📦 Installed OpenAI Baselines (`cd external/baselines && pip install -e .`)
- [ ] ⚙️ Installed project (`pip install -e .`)
- [ ] 🚀 Started training with `cd cassieWalking/exe && ./train.sh`
- [ ] 📊 Monitor training progress in console output
- [ ] 🧪 Test trained model with `./test.sh`

**Ready to train bipedal walking with pure reinforcement learning?** 🤖🚶‍♂️

Start with `./train.sh` and watch the power of end-to-end learning in action!
