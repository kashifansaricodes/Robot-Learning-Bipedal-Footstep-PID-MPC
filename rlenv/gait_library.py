import numpy as np
from utility.utility import load_dict, bezier
from scipy import interpolate
from configs.defaults import ROOT_PATH

'''
A parameterized library of walking gaits which are represented by Bezier curves.
The gait library has three dimensions: vx, vy, walking height
'''

class GaitLibrary:
    def __init__(self, secs_per_env_step):
        self.library = load_dict(ROOT_PATH + "/motions/GaitLibrary/GaitLibrary.gaitlib")
        self.secs_per_env_step = secs_per_env_step
        self.curr_stanceLeg = 1  # 1 - right leg, -1 - left leg
        self.curr_s = 0.0
        self.s_unsat_prev = 0.0
        self.t_prev = 0.0
        self.curr_ct = 0.0
        self.set_range_of_gaitparams()
        self.reset()

    def reset(self):
        self.curr_stanceLeg = np.random.choice([-1, 1], size=1, p=[0.5, 0.5])
        standing_flag = np.random.choice([True, False], p=[0.1, 0.9])
        if standing_flag:
            self.curr_s = 0.0
            self.s_unsat_prev = 0.0
        else:
            self.curr_s = np.random.uniform(0, 1)
            self.s_unsat_prev = self.curr_s
        self.t_prev = 0.0
        self.curr_ct = 0.0

    def update_gaitlib_env(self, gait_param, time_in_sec):
        _, self.curr_ct = self._get_ref_gait(gait_param, self.curr_stanceLeg)
        s_unsat = self.s_unsat_prev + (time_in_sec - self.t_prev) * self.curr_ct
        self.curr_s = min(s_unsat, 1.0005)  # walking phase, 1- end of a step
        if self.curr_s >= 1.0:  # update stanceLeg
            self.curr_stanceLeg = -1 * self.curr_stanceLeg
            self.curr_s = 0
            self.s_unsat_prev = 0
        else:
            self.s_unsat_prev = s_unsat
        self.t_prev = time_in_sec

    def _get_ref_gait(self, gait_param, stanceLeg):
        interp_point = np.clip(gait_param, self.gait_params_min, self.gait_params_max)
        if stanceLeg == 1:
            HAlpha = interpolate.interpn(
                (
                    self.library["Velocity"][0, :],
                    self.library["Velocity"][1, :],
                    self.library["Velocity"][2, :],
                ),
                self.library["RightStance_HAlpha"],
                interp_point,
                method="linear",
            )
            ct = interpolate.interpn(
                (
                    self.library["Velocity"][0, :],
                    self.library["Velocity"][1, :],
                    self.library["Velocity"][2, :],
                ),
                self.library["RightStance_ct"],
                interp_point,
                method="linear",
            )
        else:
            HAlpha = interpolate.interpn(
                (
                    self.library["Velocity"][0, :],
                    self.library["Velocity"][1, :],
                    self.library["Velocity"][2, :],
                ),
                self.library["LeftStance_HAlpha"],
                interp_point,
                method="linear",
            )
            ct = interpolate.interpn(
                (
                    self.library["Velocity"][0, :],
                    self.library["Velocity"][1, :],
                    self.library["Velocity"][2, :],
                ),
                self.library["LeftStance_ct"],
                interp_point,
                method="linear",
            )

        HAlpha = np.reshape(HAlpha, (6, 10)).T
        return HAlpha, ct  # gait alpha, 1/time of the step

    def get_ref_states(self, gait_param, look_forward=0):
        # look forward next n*self.secs_per_env_step sec states
        s_unsat = self.curr_s + look_forward * self.secs_per_env_step * self.curr_ct
        if s_unsat >= 1.0:
            stanceLeg = -1 * self.curr_stanceLeg
        else:
            stanceLeg = self.curr_stanceLeg
        s = s_unsat % 1.0
        # print(s)
        HAlpha, ct = self._get_ref_gait(gait_param, stanceLeg)
        joint_pos = bezier(HAlpha, s)
        # joint_vel = dbezier(HAlpha, s)*ct
        return joint_pos

    def set_range_of_gaitparams(self):
        self.gait_params_min = np.min(self.library["Velocity"], axis=-1)
        self.gait_params_max = np.max(self.library["Velocity"], axis=-1)

    def get_random_init_gaitparams(self):
        zero_vel_flag = np.random.choice([True, False], p=[0.1, 0.9])
        if zero_vel_flag:
            init_vx = 0.0
            init_vy = 0.0
        else:
            init_vx = np.random.uniform(
                self.gait_params_min[0], self.gait_params_max[0]
            )
            init_vy = np.random.uniform(
                self.gait_params_min[1], self.gait_params_max[1]
            )
        init_wh = np.random.uniform(0.9, 1.0)
        return np.array([init_vx, init_vy, init_wh])
