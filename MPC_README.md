# MPC Integration for Cassie Robot

This extension adds Model Predictive Control (MPC) capabilities to the existing Cassie robot simulation with reinforcement learning.

## Overview

The implementation integrates MPC with the existing reinforcement learning framework, allowing for:

1. Training RL policies with MPC as a corrective component
2. Testing trained policies with MPC enhancement
3. Comparing performance between pure RL and RL+MPC approaches

## Setup

### Install Dependencies

Make sure to install the required dependencies, including CasADi:

```bash
pip install -e .
```

### Environment Configuration

The MPC integration can be enabled through the environment configuration:

- `use_mpc`: Set to `True` to enable MPC
- `mpc_horizon`: Prediction horizon length (default: 10)
- `mpc_dt`: Time step for prediction (default: 0.01)

## Usage

### Training with MPC

To train a policy with MPC integration:

```bash
cd exe
./train_mpc.sh
```

This will train a new policy with MPC as a corrective component. The resulting model will be saved in the `ckpts/` directory.

You can modify parameters in the `train_mpc.sh` script:

```bash
mpirun -np 1 python ../scripts/train_mpc.py  --train_name 'your_model_name' \
                                           --rnd_seed 1 \
                                           --max_iters 6000 \
                                           --save_interval 100
```

### Testing with MPC

To test a policy with MPC integration:

```bash
cd exe
./test_mpc.sh
```

You can specify the model to test and whether to use MPC in the `test_mpc.sh` script:

```bash
python ../scripts/play_mpc.py --test_model 'your_model_name' --use_mpc True
```

### Using Existing Models with MPC

You can also apply MPC to existing models trained without MPC:

```bash
python ../scripts/play_mpc.py --test_model 'existing_model' --use_mpc True
```

## MPC Controller

The MPC controller is implemented in `rlenv/mpc_controller.py`. It uses:

- A simplified dynamics model for Cassie
- Optimization over a finite horizon to compute control actions
- Warm-starting to improve computational efficiency

The controller computes optimal motor torques that track reference trajectories while respecting system dynamics and constraints.

## Tuning MPC Performance

Several parameters can be adjusted to tune the MPC controller:

1. In `mpc_controller.py`:
   - Motor inertia and damping values
   - Cost function weights for different state components and controls

2. In `cassie_env.py`:
   - MPC weight to balance between RL policy and MPC controller outputs
   - Reference trajectory generation

3. In `configs/env_config.py`:
   - MPC horizon length
   - MPC time step

## Architecture

The integration follows this architecture:

1. RL policy produces initial control actions
2. MPC controller computes optimal control over a horizon
3. Final control is a weighted combination of RL and MPC outputs
4. Environment applies the control and provides observation for next step

This hybrid approach leverages the strengths of both RL (learning complex behaviors) and MPC (optimal control with dynamics constraints). 