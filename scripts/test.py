from baselines.common import tf_util as U
import tensorflow as tf
from mujoco_env.cassie_env import CassieEnv
import ppo.policies as policies
from configs.env_config import config_play
from configs.defaults import ROOT_PATH
import mujoco_env

# to ignore specific deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse

model_folder = ROOT_PATH + "/ckpts/"

def get_args():
    parser = argparse.ArgumentParser(description="testing_setup")

    parser.add_argument(
        "--test_model", type=str, default=None, help="checkpoint to test (default None)"
    )
    parser.add_argument(
        "--test_episode_len", type=int, default=20000, help="episode length to test -- can be longer than the env episode length"
    )
    args = parser.parse_args()

    return args


args = get_args()
print(mujoco_env.__file__)
# model name to test
test_model = args.test_model
test_episode_len = args.test_episode_len


def main():
    """
    restore latest model from ckpt
    """
    
    env = CassieEnv(config=config_play)
    model_dir = model_folder + test_model
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    model_path = latest_checkpoint
    config = tf.ConfigProto(device_count={"GPU": 0})

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

    U.make_session(config=config)
    U.load_state(model_path)

    draw_state = env.render()
    while draw_state:
        obs_vf, obs_pol = env.reset()
        for _ in range(test_episode_len):
            if not env.vis.ispaused():
                ac = pi.act(stochastic=False, ob_vf=obs_vf, ob_pol=obs_pol)[0]
                obs_vf, obs_pol, reward, done, info = env.step(ac)
                draw_state = env.render()
                if done:
                    env.reset()
                    break
            else:
                while env.vis.ispaused() and draw_state:
                    draw_state = env.render()


if __name__ == "__main__":
    main()
