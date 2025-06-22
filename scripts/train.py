import os
from baselines.common import tf_util as U
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from ppo.policies import MLPCNNPolicy
from mujoco_env.cassie_env import CassieEnv
from configs.defaults import ROOT_PATH
from baselines import logger
from mpi4py import MPI
import argparse

os.environ["OPENAI_LOG_FORMAT"] = "stdout,log,tensorboard"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_folder = ROOT_PATH + "/ckpts/"


def get_args():
    parser = argparse.ArgumentParser(description="training_setup")

    parser.add_argument(
        "--train_name", type=str, default="test", help="naming the training"
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
# NOTE: NOT explicitly setting rnd seed -- it will be randomized every time the script started
# NOTE: if explicitly setting rnd seed is needed, should be careful with the MPI -- it should be different for each MPI runner
rnd_seed = args.rnd_seed 
# max iteration for training 
max_iters = args.max_iters
print(max_iters)
# how frequent the ckpt will be saved
save_interval = args.save_interval

# define the name
if restore_cont and restore_from:
    saved_model = train_name + "_rnds" + str(rnd_seed) + "_cont" + str(restore_cont)
else:
    saved_model = train_name + "_rnds" + str(rnd_seed)

restore_model_from_file = restore_from
os.environ["OPENAI_LOGDIR"] = ROOT_PATH + "/logs/" + saved_model

print("[Train]: MODEL_TO_SAVE", saved_model)


def train(max_iters, with_gpu=True, callback=None):
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

    from configs.env_config import config_train

    env = CassieEnv(config=config_train)

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
    train(max_iters=max_iters, callback=training_callback)
