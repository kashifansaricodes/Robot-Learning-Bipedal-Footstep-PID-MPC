#!/bin/bash

MODEL="new_training_rnds1"
if [ "$1" != "" ]; then
  MODEL="$1"
fi

# Export necessary paths
export PYTHONPATH=/home/abhishek/Bi_Pedal_Robot-Learning

# Set library preload (important for MuJoCo)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Use software rendering (for systems with problematic GPU drivers)
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

# Try different MuJoCo rendering backends
# Options: 'glfw' (default), 'egl', 'osmesa', or 'glx'
# Uncomment one of these if the default doesn't work
export MUJOCO_GL=egl
# export MUJOCO_GL=osmesa
# export MUJOCO_GL=glx

# Get the project root directory (parent of the exe directory)
PROJECT_ROOT="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

# Run the script using the absolute path
python ${PROJECT_ROOT}/scripts/play_mpc_model.py --model_name $MODEL --episode_len 10000 $2 $3 $4 