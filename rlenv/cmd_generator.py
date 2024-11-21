import numpy as np
from utility.utility import local2global_yawonly
from scipy import interpolate

'''
To generate randomized/fixed command (vx, vy, walking height, turning yaw rate) 
with a schedule with the progress of the episode. 
This will be used in reference_generator.py
'''

class CmdGenerateor:
    def __init__(self, env_max_timesteps, secs_per_env_step, config):
        self.env_max_timesteps = env_max_timesteps
        self.fix_gait_timesteps = 200  # the elapsed time for the very first cmd, turned out to be useless though
        if config["fixed_gait"]:
            self.fix_gait_allsteps = True  # NOTE: set true to train a fixed gait
            if config["fixed_gait_cmd"]:
                self.fix_gait = np.array(config["fixed_gait_cmd"][:3]).reshape((3, 1))
                self.fix_rot = np.radians(
                    np.array([0.0, 0.0, config["fixed_gait_cmd"][3]]).reshape((3, 1))
                )
                self.add_rotation = False  # NOTE: set true to train rotation
            else:
                print("[Error]: no cmd for the fixed gait is specifid!")
                raise NotImplementedError
        else:
            self.fix_gait_allsteps = False  # NOTE: set true to train a fixed gait
            self.add_rotation = config[
                "add_rotation"
            ]  # NOTE: set true to train rotation
        self.env_freq = 1 / secs_per_env_step
        self.secs_per_step = secs_per_env_step
        self.time_ref = None
        self.cmds_ref = None
        self.rot_time_ref = None
        self.rot_ref = None
        self.interp_ref_cmds_func = None
        self.interp_ref_rot_func = None
        self.__init_cmd_range()
        self.reset()

    def reset(self):
        self.in_stand_mode = False
        self.base_ref_xy_global = np.array([0.0, 0.0])
        self.base_ref_rot = np.array([0.0, 0.0, 0.0])
        self.last_ref_vxy_global = np.array([0.0, 0.0])
        self.last_ref_vyaw = 0.0
        self.num_tp = np.random.randint(5, 12)
        self.num_tp_rot = np.random.randint(5, 12)
        self.shuffle_cmds()
        self.update_cmd_env(time_in_sec=0)

    def update_cmd_env(self, time_in_sec):
        self._update_ref_base()
        self._update_ref_cmds(time_in_sec)
        self.curr_time = time_in_sec

    def _update_ref_base(self):
        self.__update_ref_base_pos_global()
        self.__update_ref_base_rot()

    def _update_ref_cmds(self, time_in_sec):
        if not self.in_stand_mode:
            self.ref_gait_params = self.interp_ref_cmds_func(time_in_sec)
            self.ref_rot_cmds = self.interp_ref_rot_func(time_in_sec)
        else:
            pass  # overwrite the ref gait cmds by outside cmd

    def shuffle_cmds(self):
        vel_x = np.random.uniform(self.cmds_min[0], self.cmds_max[0], self.num_tp)
        vel_y = np.random.uniform(self.cmds_min[1], self.cmds_max[1], self.num_tp)
        pos_z = np.random.uniform(self.cmds_min[2], self.cmds_max[2], self.num_tp)

        roll = np.radians(
            np.random.uniform(self.rot_min[0], self.rot_max[0], self.num_tp_rot)
        )
        pitch = np.radians(
            np.random.uniform(self.rot_min[1], self.rot_max[1], self.num_tp_rot)
        )
        yaw_vel = np.radians(
            np.random.uniform(self.rot_min[2], self.rot_max[2], self.num_tp_rot)
        )

        # 10% to include a step in place gait
        step_in_place_flag = np.random.choice([False, True], p=[0.9, 0.1])
        if step_in_place_flag:
            step_in_place_idx = np.random.randint(0, self.num_tp)
            vel_x[step_in_place_idx] = 0.0
            vel_y[step_in_place_idx] = 0.0
            pos_z[step_in_place_idx] = np.random.uniform(
                self.cmds_min[2], self.cmds_max[2]
            )

        zero_rot = np.random.choice([True, False], p=[0.1, 0.9])
        if zero_rot:
            zero_rot_idx = np.random.randint(0, self.num_tp_rot)
            roll[zero_rot_idx] = 0.0
            pitch[zero_rot_idx] = 0.0
            yaw_vel[zero_rot_idx] = 0.0

        init_vx = np.random.uniform(-1.0, 1.0)
        init_vy = np.random.uniform(-0.3, 0.3)
        init_z = np.random.uniform(0.9, 1.0)
        init_cmds = np.array([init_vx, init_vy, init_z]).reshape(
            (3, 1)
        )  # init from a random walking forward cmds
        init_yaw_vel = np.radians(0.0)
        init_pitch = np.radians(0.0)
        init_roll = np.radians(0.0)
        init_rot = np.array([init_roll, init_pitch, init_yaw_vel]).reshape((3, 1))

        if not self.fix_gait_allsteps:
            self.cmds_ref = np.hstack([init_cmds, np.vstack([vel_x, vel_y, pos_z])])
            self.rot_ref = np.hstack([init_rot, np.vstack([roll, pitch, yaw_vel])])
            # first fix_gait_timesteps: constant gait, rest: change in larger size
            self.time_ref = np.hstack(
                [
                    0,
                    np.linspace(
                        self.fix_gait_timesteps / self.env_freq,
                        self.env_max_timesteps / self.env_freq + 2,
                        self.num_tp,
                    ),
                ]
            )
            self.rot_time_ref = np.hstack(
                [
                    0.0,
                    np.linspace(
                        self.fix_gait_timesteps / self.env_freq,
                        self.env_max_timesteps / self.env_freq + 2,
                        self.num_tp_rot,
                    ),
                ]
            )
            self.ref_gait_params = np.array([init_vx, init_vy, init_z])
            self.ref_rot_cmds = np.array([init_roll, init_pitch, init_yaw_vel])
        else:
            self.cmds_ref = np.hstack([self.fix_gait, self.fix_gait])
            self.rot_ref = np.hstack([self.fix_rot, self.fix_rot])
            self.time_ref = np.hstack([0, self.env_max_timesteps / self.env_freq + 2])
            self.rot_time_ref = np.hstack(
                [0, self.env_max_timesteps / self.env_freq + 2]
            )
            self.ref_gait_params = self.fix_gait.flatten()
            self.ref_rot_cmds = self.fix_rot.flatten()

        self.interp_ref_cmds_func = interpolate.interp1d(
            self.time_ref, self.cmds_ref, kind="previous", axis=-1
        )
        self.interp_ref_rot_func = interpolate.interp1d(
            self.rot_time_ref, self.rot_ref, kind="previous", axis=-1
        )

    @property
    def curr_ref_gaitparams(self):
        # output: vx, vy, walking height
        return self.ref_gait_params.copy()

    @property
    def curr_ref_rotcmds(self):
        # output: roll angle, pitch angle, yaw_vel
        return self.ref_rot_cmds.copy()

    def clear_stand_mode(self):
        self.in_stand_mode = False

    def start_stand_mode(self):
        self.in_stand_mode = True
        self.ref_gait_params = np.array([0.0, 0.0, 0.95])
        self.ref_rot_cmds = np.array([0.0, 0.0, 0.0])

    def get_ref_base_global(self):
        return self.base_ref_xy_global, self.base_ref_rot

    def set_ref_global_pos(self, xy):
        self.base_ref_xy_global = xy

    def set_ref_global_yaw(self, yaw):
        self.base_ref_rot[2] = yaw

    def __update_ref_base_pos_global(self):
        # NOTE: gait library is giving local ref velocity and rotation in the base frame
        ref_gait_params = self.curr_ref_gaitparams
        ref_vxy_global = local2global_yawonly(
            ref_gait_params[:2], self.base_ref_rot[-1]
        )
        self.base_ref_xy_global += (
            (ref_vxy_global + self.last_ref_vxy_global) / 2.0 * self.secs_per_step
        )
        self.last_ref_vxy_global = ref_vxy_global

    def __update_ref_base_rot(self):
        ref_rot_cmds = self.curr_ref_rotcmds
        base_ref_yaw = self.base_ref_rot[2]
        # ref_rot [2] is yaw vel
        self.base_ref_rot[[0, 1]] = ref_rot_cmds[[0, 1]]
        self.base_ref_rot[2] = (
            base_ref_yaw
            + (ref_rot_cmds[2] + self.last_ref_vyaw) / 2.0 * self.secs_per_step
            + np.pi
        ) % (2 * np.pi) - np.pi
        # print("ref yaw:{}".format(np.degrees(self.base_ref_rot[2])))
        self.last_ref_vyaw = ref_rot_cmds[2]

    def __init_cmd_range(self):
        self.cmds_max = [1.5, 0.6, 1.0]  # vx vy wh
        self.cmds_min = [-1.5, -0.6, 0.65]
        if self.add_rotation:
            self.rot_min = np.array([0.0, 0.0, -45.0]).reshape(
                (3, 1)
            )  # roll pitch yaw_vel
            self.rot_max = np.array([0.0, 0.0, 45.0]).reshape((3, 1))
        else:
            self.rot_min = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
            self.rot_max = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
