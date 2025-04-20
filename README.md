# Bi_Pedal_Robot-Robot-Learning


This README outlines all the steps taken to set up and run the project with MuJoCo 2.1.0, TensorFlow 1.15, and OpenAI Baselines in a Conda environment on Ubuntu 22.04.

---

file:///home/kashif/Pictures/Screenshots/Screenshot%20from%202025-04-20%2018-27-04.png

---

## ✅ Requirements

- Ubuntu 22.04
- NVIDIA/Intel GPU (or software rendering)
- MuJoCo 2.1.0
- Conda (Miniconda or Anaconda)

---

## ⚙️ Step-by-Step Setup

### 1. Create Conda Environment

```bash
conda create -n cassie python=3.7.16 -y
conda activate cassie
```

### 2. Install Dependencies

```bash
pip install tensorflow==1.15
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..
pip install gym
```

### 3. Install `mujoco-py`

```bash
pip install mujoco-py==2.1.2.14
```

### 4. Install MuJoCo 2.1.0

1. Download MuJoCo 2.1.0 from: https://mujoco.org/download  
2. Extract it:

```bash
mkdir -p ~/.mujoco
tar -xzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
mv ~/.mujoco/mujoco210-linux-x86_64 ~/.mujoco/mujoco210
```

### 5. Set Environment Variables

Add the following to your `~/.bashrc`:

```bash
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_GL=egl  # Optional: for headless rendering
```

Then:
```bash
source ~/.bashrc
```

### 6. Install System Dependencies

```bash
sudo apt update
sudo apt install libgl1-mesa-glx libgl1-mesa-dri libglfw3 libosmesa6 libglew-dev mesa-utils
```

### 7. Fix GLIBCXX Compatibility Issue

MuJoCo’s OpenGL backend may require a newer `libstdc++.so.6` than Conda provides.

**Temporary fix:**
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

**(Optional)** To make this persistent:

```bash
mkdir -p ~/miniconda3/envs/cassie/etc/conda/activate.d
nano ~/miniconda3/envs/cassie/etc/conda/activate.d/env_vars.sh
```

Add inside `env_vars.sh`:
```bash
#!/bin/bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

Then:
```bash
chmod +x ~/miniconda3/envs/cassie/etc/conda/activate.d/env_vars.sh
```

---

## 🚀 Running the Program

Navigate to the execution directory:

```bash
cd cassie_rl_walking/exe
./test_static.sh
```

```bash
cd cassie_rl_walking/exe
./play.sh
```

If you still face `GLEW initialization` errors:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
./play.sh
```

---

## ✅ Final Notes

- Ensure `~/.mujoco/mujoco210/bin/` contains `libmujoco210nogl.so` and other required `.so` files.
- `MUJOCO_GL=egl` is suitable for headless or offscreen rendering setups.
- This setup supports both GUI and non-GUI execution of MuJoCo-based RL environments.

