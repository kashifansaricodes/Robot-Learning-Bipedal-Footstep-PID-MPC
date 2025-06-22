## Reinforcement Learning for Bipedal(Cassie) Locomotion

This folder contains the code implementation for my project

Team Member: Sounderya Varagur Venugopal (UID: 121272423)

### Installation

1. Install MuJoCo:
  - Download MuJoCo **210** from [HERE](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0)
  - Extract the downloaded mujoco210 directory into `~/.mujoco/mujoco210`.
  - Add this to bashrc or export: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin`

2. Extract the contents of the zip folder and navigate into the directory

3. Create an environment, `cassie_walking`, for python<3.8:

  - `conda create -n cassie-rl python=3.7`
  - `conda activate cassie-rl`


4. Install the dependencies and setup the enviornments
  - `pip install -e .` (it will install tensorflow=1.15)

5. Install [openai baselines](https://github.com/openai/baselines)
  - `cd external/baselines`
  - `pip install -e .`
  - `cd ../..`


### Running the code 

1. Train a policy:

    - `cd cassieWalking/exe`
    - `./train.sh`

    The parameters are set in `cassieWalking/exe/train.sh`. In `mpirun -np xx python`, `xx` is the number of workers (CPU core) to be used. The results will be stored in the `ckpts` folder

2. Test a policy:

    - `cd cassieWalking/exe`
    - `./test.sh`

    The model to be tested is set in `--test_model` in `./test.sh`. 


## References
- [cassie-mujoco-sim](https://github.com/osudrl/cassie-mujoco-sim)
- [LearningHumanoidWalking](https://github.com/rohanpsingh/LearningHumanoidWalking)
- [apex](https://github.com/osudrl/apex)

