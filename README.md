## Cassie RL for Versatile Walking Skill
This repository is an example code for bipedal locomotion control using reinforcement learning introduced in the paper [''Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control''](https://arxiv.org/abs/2401.16889). 
This uses a person-sized bipedal robot, [Cassie](https://agilityrobotics.com/content/2017/agility-robotics-introduces-cassie-a-dynamic-and-talented-robot-delivery-ostrich-nfxym-bjn8y-ake95), as an example, and a checkpoint of trained walking controller for Cassie is also provided for testing and evaluation in simulation. 

The policy can control the robot to track varying commands in [sagittal_velocity, lateral_velocity, walking_height, turning_yaw_rate] and also switching between walking and standing.

**Warning:** **Although the provided checkpoint can be transferred to Cassie's hardware, deploying it is at your own risk. We do not assume responsibility for any potential damage.**

**Special Note:** This is an "old" codebase for RL locomotion control for Cassie — it was developed before the era of GPU-accelerated simulators in RL. It is built on a CPU-based simulator and a CPU-based learning framework. The good thing is that it doesn't require a high-end GPU to run. However, training it requires a CPU server (>16 cores) and is quite slow. 

>**Why not GPU?** While training can utilize a GPU, **accurately** simulating Cassie with a closed-loop kinematic chain, including the passive spring-driven tarsus and shin joints, is currently best supported in MuJoCo, which was CPU-based at the time. Copying simulated data from the CPU to the GPU is also costly. Therefore, this codebase is CPU-centered. 


## Getting Started

### Installation

We test our codes under the following environment:

- Ubuntu 20.04
- Python 3.7.16
- Tensorflow 1.15
- MuJoCo 2.1.0
- The RL library is [openai baselines](https://github.com/openai/baselines).

1. Install MuJoCo:
  - Download MuJoCo **210** from [HERE](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0)
  - Extract the downloaded mujoco210 directory into `~/.mujoco/mujoco210`.
  - Add this to bashrc or export: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin`

2. Create an environment, say named `cassie-rl`, for python<3.8:

  - `conda create -n cassie-rl python=3.7`
  - `conda activate cassie-rl`

3. Clone this repository.

  - `git clone --recurse-submodules https://github.com/HybridRobotics/cassie_rl_walking.git`
  - `cd cassie_rl_walking`

4. Install the dependencies and setup the enviornments
  - `pip install -e .` (it will install tensorflow=1.15)

5. Install [openai baselines](https://github.com/openai/baselines) which is included as a submodule.
  - `cd external/baselines`
  - `pip install -e .`
  - `cd ../..`


### Tutorial
0. Environment configurations

    You can play around the parameters for the environment in `configs/env_config.py`. The parameters in this config file will be passed to the environment. 

    For example, to test a single fixed command, set:

    - `"fixed_gait": True`
    - change the command you want to test in `"fixed_gait_cmd"`

    To test all range of command, set:

    - `"fixed_gait": False`
    - `"add_rotation": True` to also include turning command

    To include standing during episode, set:
    - `"add_standing": True`

    To add dynamics randomization, set:

    - `"minimal_rand": False` to add full range of dynamics randomization
    - `is_noisy: True` to add noise to observation
    - `add_perturbation: False` to add external perturbation wrench to the robot base (pelvis)


1. Executables are under `cassie_rl_walking/exe`

2. Test a policy:

    - `cd cassie_rl_walking/exe`
    - `./play.sh`

    Change the `--test_model` in `./play.sh` to the model you want to test.
    You can play around the environemt configs as descrived before 


3. Train a policy:

    - `cd cassie_rl_walking/exe`
    - `./train.sh`

    You need to set the arguments in `cassie_rl_walking/exe/train.sh`
    
    **Note:** It uses [MPI](https://mpi4py.readthedocs.io/en/stable/) for multithread training. In `train.sh` change `mpirun -np 1 python` to `mpirun -np xx python` where `xx` is the number of workers (CPU core) you want to use. `16` was used as default, if you have a CPU server with >16 cores. The PPO batch size is `num_of_workers x timesteps_per_actorbatch`, and `timesteps_per_actorbatch` is set to `4096` in `cassie_rl_walking/scripts/train.py`. 
    
    You can set the environemt configs to train as descrived before
    
4. Visualize the reference motion (no dynamics simulation):

    - `cd cassie_rl_walking/exe`
    - `./test_static.sh`

    You can play around the environemt configs as descrived before 

## Citation

If you find our work helpful, please cite:

```bibtex
@article{li2024reinforcement,
  title={Reinforcement learning for versatile, dynamic, and robust bipedal locomotion control},
  author={Li, Zhongyu and Peng, Xue Bin and Abbeel, Pieter and Levine, Sergey and Berseth, Glen and Sreenath, Koushil},
  journal={The International Journal of Robotics Research},
  year={2024},
  publisher={SAGE Publications Sage UK: London, England}
}
```

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Acknowledgements
- [cassie-mujoco-sim](https://github.com/osudrl/cassie-mujoco-sim): Our codebase is built upon the simulation library for Cassie robot using MuJoCo.

- [Xuxin Cheng](https://chengxuxin.github.io/): Xuxin is an inital developer of this work. If you find an MLP (short history only) is useful enough, please consider to cite:
    ```bibtex
    @inproceedings{li2021reinforcement,
    title={Reinforcement learning for robust parameterized locomotion control of bipedal robots},
    author={Li, Zhongyu and Cheng, Xuxin and Peng, Xue Bin and Abbeel, Pieter and Levine, Sergey and Berseth, Glen and Sreenath, Koushil},
    booktitle={International Conference on Robotics and Automation (ICRA)},
    pages={2811--2817},
    year={2021}
    }
    ```
