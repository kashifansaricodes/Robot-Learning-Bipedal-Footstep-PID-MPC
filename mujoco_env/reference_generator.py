import numpy as np
import math
from transforms3d.euler import euler2quat as euler_to_quaternion
from mujoco_env.gait_library import GaitLibrary
from mujoco_env.cmd_generator import CmdGenerateor
from configs.defaults import STANDING_POSE

'''
A generator for reference motion, which will be used in cassie_env.py. 
The env script only look at the ref_dict returned from this class. 
ref_dict contains all the info of the refernece motion, 
including reference motor position, motor velocity, base translational/rotational position/velocity
The ReferenceGenerator composes the data from cmd_generator.py & gait_libraray.py
and randomly set the robot to stand for a random timespan. 
'''

class ReferenceGenerator:
    def __init__(
        self,
        env_max_timesteps,
        secs_per_env_step,
        config,
    ):
        # print("Here")
        self.cmd_generator = CmdGenerateor(env_max_timesteps, secs_per_env_step, config)
        self.gait_library = GaitLibrary(secs_per_env_step)
        self.time_stand_transit_cooling = 3.0  # allow 3 sec to transit to standing
        self.norminal_standing = np.copy(STANDING_POSE)
        self.add_standing = config["add_standing"]

        # Footstep trajectory additions
        self.use_footstep_plan = config.get("use_footstep_plan", True)
        self.footstep_traj = []
        self.footstep_index = 0
        self.dt = secs_per_env_step

        self.reset()

    def reset(self):
        self.time_in_sec = 0.0
        self.cmd_generator.reset()
        self.gait_library.reset()
        if self.add_standing:
            stand_flag = np.random.choice([True, False], p=[0.9, 0.1])
            if stand_flag:
                self.time_standing_start = np.random.uniform(5.0, 30.0)
            else:
                self.time_standing_start = 10000.0
        else:
            self.time_standing_start = 10000.0
        self.standing_span = np.random.uniform(2.0, 30.0)
        self.start_standing = False
        self.end_standing = False
        self.last_ref_gaitparams = np.array([0.0, 0.0, 0.98])
        self.last_ref_rotparams = np.array([0.0, 0.0, 0.0])
        init_stand_flag = np.random.choice([True, False], p=[0.5, 0.5])
        if init_stand_flag:
            self.init_standing_flag = True  # stand at the first time
            self.last_standing_flag = False
        else:
            self.init_standing_flag = False  # jump at the first time
            self.last_standing_flag = True

        if self.use_footstep_plan:
            plans = load_footstep_plans("../mujoco_env/footstep_plans.txt")
            # print("Hereeeeee")
            raw_steps = plans[0]
            self.footstep_traj = interpolate_trajectory(raw_steps, dt=self.dt)
            self.footstep_index = 0

    def update_ref_env(self, time_in_sec, base_xy_g, base_yaw):
        self.time_in_sec = time_in_sec

        if self.use_footstep_plan and self.footstep_index < len(self.footstep_traj):
            x, y, theta = self.footstep_traj[self.footstep_index]
            self.last_ref_gaitparams = np.array([0.0, 0.0, 0.9])
            self.last_ref_rotparams = np.array([0.0, 0.0, theta])
            self.footstep_index += 1
            return

        if (
            not self.start_standing
            and self.time_in_sec >= self.time_standing_start
            and self.time_in_sec < self.time_standing_start + self.standing_span
        ):
            self.start_standing = True
        if (
            self.start_standing
            and self.time_in_sec >= self.time_standing_start + self.standing_span
        ):
            self.end_standing = True
            self.start_standing = False
            self.cmd_generator.clear_stand_mode()
        if self.start_standing:
            self.cmd_generator.start_stand_mode()
        self.cmd_generator.update_cmd_env(time_in_sec)
        ref_gaitparams = self.cmd_generator.curr_ref_gaitparams
        ref_rotparams = self.cmd_generator.curr_ref_rotcmds
        self.gait_library.update_gaitlib_env(
            gait_param=ref_gaitparams, time_in_sec=time_in_sec
        )
        if abs(self.last_ref_gaitparams[0] - ref_gaitparams[0]) >= 0.01:
            self.cmd_generator.set_ref_global_pos(xy=base_xy_g)
        if abs(self.last_ref_rotparams[-1] - ref_rotparams[-1]) >= 0.002:
            self.cmd_generator.set_ref_global_yaw(yaw=base_yaw)
        self.last_ref_gaitparams = ref_gaitparams
        self.last_ref_rotparams = ref_rotparams

    def get_init_pose(self):
        init_gait_params = self.gait_library.get_random_init_gaitparams()
        ref_mpos = self.gait_library.get_ref_states(init_gait_params)
        ref_base_pos_from_cmd, _ = self.cmd_generator.get_ref_base_global()
        ref_base_pos = np.array(
            [
                ref_base_pos_from_cmd[0],
                ref_base_pos_from_cmd[1],
                init_gait_params[-1],
            ]
        )
        norminal_pose_flag = np.random.choice([True, False], p=[0.5, 0.5])
        ref_base_rot = np.array(
            [
                np.radians(np.random.uniform(-2.0, 2.0)),
                np.radians(np.random.uniform(-5.0, 5.0)),
                np.radians(np.random.uniform(-10.0, 10.0)),
            ]
        ).reshape((1, 3))
        if norminal_pose_flag:
            ref_base_pos, _, ref_mpos = self.norminal_pose
            stand_abduction_offset = np.radians(np.random.uniform(-1.0, 7.5, (2,)))
            ref_mpos[0] = ref_mpos[0] + stand_abduction_offset[0]
            ref_mpos[5] = ref_mpos[5] - stand_abduction_offset[1]
            stand_knee_offset = np.radians(np.random.uniform(-5.0, 5.0, (2,)))
            stand_thigh_offset = np.radians(np.random.uniform(-5.0, 5.0, (2,)))
            ref_mpos[3] += stand_knee_offset[0]
            ref_mpos[3 + 5] += stand_knee_offset[1]
            ref_mpos[2] += stand_thigh_offset[0]
            ref_mpos[2 + 5] += stand_thigh_offset[1]
            ref_base_pos[2] += np.random.uniform(-0.05, 0.05)
        return ref_base_pos, ref_base_rot, ref_mpos

    def get_ref_motion(self, look_forward=0):
        ref_dict = dict()
        ref_gait_params = self.last_ref_gaitparams
        ref_rot_params = self.last_ref_rotparams
        ref_base_pos_from_cmd, ref_base_rot_from_cmd = self.cmd_generator.get_ref_base_global()
        ref_mpos = self.gait_library.get_ref_states(ref_gait_params, look_forward)
        ref_dict["base_pos_global"] = np.array(
            [*ref_base_pos_from_cmd, ref_gait_params[-1]]
        )
        ref_dict["base_rot_global"] = ref_base_rot_from_cmd
        ref_dict["base_vel_local"] = np.array(
            [ref_gait_params[0], ref_gait_params[1], ref_rot_params[-1]]
        )
        if self.start_standing:
            ref_dict["motor_pos"] = self.norminal_mpos
            ref_dict["motor_vel"] = np.zeros((10,))
        else:
            ref_dict["motor_pos"] = ref_mpos
            ref_dict["motor_vel"] = np.zeros((10,))
        return ref_dict

    def get_curr_params(self):
        return self.last_ref_gaitparams, self.last_ref_rotparams

    @property
    def norminal_pose(self):
        return self.norminal_base_pos, self.norminal_base_rot, self.norminal_mpos

    @property
    def norminal_base_pos(self):
        return np.copy(self.norminal_standing[[0, 1, 2]])

    @property
    def norminal_base_rot(self):
        return np.copy(self.norminal_standing[[3, 4, 5]])

    @property
    def norminal_mpos(self):
        return np.copy(self.norminal_standing[6:])

    @property
    def in_transit_to_stand(self):
        return (
            self.start_standing
            and self.time_in_sec <= self.time_standing_start + self.time_stand_transit_cooling
        )

    @property
    def in_stand_mode(self):
        return self.start_standing

def load_footstep_plans(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    plans = []
    for block in content.strip().split('---'):
        steps = []
        for line in block.strip().splitlines():
            if line:
                x, y, theta = map(float, line.strip().split(','))
                steps.append((x, y, theta))
        if steps:
            plans.append(steps)
    return plans

def interpolate_trajectory(footsteps, dt=0.02, step_duration=0.5):
    ref_traj = []
    for i in range(len(footsteps) - 1):
        x0, y0, th0 = footsteps[i]
        x1, y1, th1 = footsteps[i + 1]
        steps = int(step_duration / dt)
        for t in range(steps):
            alpha = t / steps
            x = (1 - alpha) * x0 + alpha * x1
            y = (1 - alpha) * y0 + alpha * y1
            theta = (1 - alpha) * th0 + alpha * th1
            ref_traj.append((x, y, theta))
    ref_traj.append(footsteps[-1])
    return ref_traj
