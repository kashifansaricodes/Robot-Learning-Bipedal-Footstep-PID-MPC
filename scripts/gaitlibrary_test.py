from rlenv.cassie_env import CassieEnv
import numpy as np
from utility.utility import euler2quat
from configs.env_config import config_static


env = CassieEnv(config=config_static)
obs = env.reset()

draw_state = env.render()
count = 0
while draw_state:
    if not env.vis.ispaused():
        draw_state = env.render()
        ref_gait_params, ref_rot_params = env.reference_generator.get_curr_params()
        ref_q = euler2quat(env.ref_dict["base_rot_global"])
        sim_rot = np.array(env.sim.qpos()[3:7])
        ref_base = np.hstack(
            [env.ref_dict["base_pos_global"].reshape((1, 3)), ref_q.reshape((1, 4))]
        )
        env.set_motor_base_pos(motor_pos=env.ref_dict["motor_pos"], base_pos=ref_base)
        acs_norm = env.acs_actual2norm(env.ref_dict["motor_pos"])
        obs_vf, obs_pol, reward, done, info = env.step(acs_norm)

        count += 1
        if count >= config_static["max_timesteps"]:
            env.reset()
            count = 0

    else:
        while env.vis.ispaused() and draw_state:
            draw_state = env.render()
