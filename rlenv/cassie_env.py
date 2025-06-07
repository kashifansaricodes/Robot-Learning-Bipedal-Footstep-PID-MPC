from rlenv.cassiemujoco import *
from rlenv.cassiemujoco_ctypes import *
from rlenv.reference_generator import ReferenceGenerator
from rlenv.action_filter import ActionFilterButter
from rlenv.env_randomlizer import EnvRandomlizer
from rlenv.perturbation_generator import PerturbationGenerator
from utility.cassie_fk import CassieFK
from gym import spaces
from configs.defaults import *
from utility.utility import *
from collections import deque


class CassieEnv:
    def __init__(
        self,
        config,
    ):
        self.max_timesteps = config["max_timesteps"]
        self.action_bounds = np.copy(ACTUATED_JOINT_RANGE)
        self.safe_action_bounds = np.copy(self.action_bounds)
        # for safe joint limits on hardware
        self.safe_action_bounds[:, [0, 5]] *= 0.5
        self.safe_action_bounds[:, [1, 6]] *= 0.8
        self.base_idx = [0, 1, 2, 3, 4, 5, 6]
        self.motor_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.motor_vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.pGain = np.copy(DEFAULT_PGAIN)
        self.dGain = np.copy(DEFAULT_DGAIN)
        self.offset_footJoint2midFoot = np.sqrt(0.01762**2 + 0.05219**2)
        self.sim = CassieSim(ROBOT_MODEL_FILE)
        if config["is_visual"]:
            self.is_visual = True
            self.vis = CassieVis(self.sim, ROBOT_MODEL_FILE)
            if config["cam_track_robot"]:
                self.vis.set_cam("cassie-pelvis", 2.5, 90, 0)
        else:
            self.is_visual = False
        self.step_zerotorque = config["step_zerotorque"]
        self.cassie_fk = CassieFK()
        # init variables
        self.qpos = np.copy(self.sim.qpos())
        self.qvel = np.copy(self.sim.qvel())
        self.qacc = np.copy(self.sim.qacc())
        self.obs_cassie_state = state_out_t()
        self.cassie_out = cassie_out_t()
        # Initialize MPC controller if specified
        self.use_mpc = config.get("use_mpc", False)
        if self.use_mpc:
            from rlenv.mpc_controller import CassieMPC
            self.mpc_horizon = config.get("mpc_horizon", 10)
            self.mpc_dt = config.get("mpc_dt", 0.01)
            self.mpc_controller = CassieMPC(
                horizon=self.mpc_horizon,
                dt=self.mpc_dt,
                num_motors=10  # Cassie has 10 motors
            )
        self.u = self._init_u()
        self.num_motor = 10
        self.sim_freq = 2000
        self.appx_env_freq = 30  # 30 hz but the real control frequency is not exactly 30Hz because we round up the num_sims_per_env_step
        self.num_sims_per_env_step = self.sim_freq // self.appx_env_freq
        self.secs_per_env_step = self.num_sims_per_env_step / self.sim_freq
        self.real_env_freq = int(1 / self.secs_per_env_step)
        self.history_len_vf = 4
        self.history_len_pol = 2 * self.real_env_freq
        # observation
        self.observation_space_pol = None
        self.observation_space_vf = None 
        self.observation_space_pol_cnn = None
        self.extrinsics_dim = None
        # the low & high does not actually limit the actions output from MLP network, manually clip instead
        self.action_space = spaces.Box(low=-100, high=100, shape=(self.num_motor,))
        self.previous_obs = deque(maxlen=self.history_len_vf)
        self.previous_acs = deque(maxlen=self.history_len_vf)
        self.long_history = deque(maxlen=self.history_len_pol)
        # reference motion
        self.reference_generator = ReferenceGenerator(
            env_max_timesteps=self.max_timesteps,
            secs_per_env_step=self.secs_per_env_step,
            config=config,
        )
        # action filter
        self.action_filter_order = 2
        self.use_action_filter = config.get("use_action_filter", True)  # Default to True for backward compatibility
        self.__init_action_filter()
        # dynamics randomization
        self.__set_env_type(config)
        # set reward
        self.__init_reward_func()
        self.__init_env_randomlizer()
        self.__init_perturbation_generator()
        # init step
        self.__init_step_func()
        # reset and init others
        self.reset()
        # self.sim.set_geom_name_pos('table', [0.6, 0.85, -0.265]) # 0.5 is the max

    ##########################################
    #            Init and Reset              #
    ##########################################
    def __set_env_type(self, config):
        # NOTE: set true to use minimal rand range -> only rand floor friction
        self.minimal_rand = config["minimal_rand"]
        # NOTE: set true to add noise on observation
        self.noisy = config["is_noisy"]  
        # NOTE: set true to add external perturbation
        self.perturbation = config["add_perturbation"]

    def __init_reward_func(self):
        self.reward_names = [
            "r_mpos",
            "r_mvel",
            "r_ptrans",
            "r_ptrans_velx",
            "r_ptrans_vely",
            "r_prot",
            "r_prot_vel",
            "r_ptrans_z",
            "r_torque",
            "r_foot_force",
            "r_acc",
            "r_footpos",
            "r_delta_acs",
        ]
        w_mpos = 15.0
        w_mvel_nonstand = 0.0
        w_mvel_stand = 15.0
        w_ptrans_pos = 6.0
        w_ptrans_velx = 7.5
        w_ptrans_vely = 7.5
        w_prot_pos = 12.5
        w_prot_vel = 3.0
        w_ptrans_z = 5.0
        w_torque = 3.0
        w_foot_force = 10.0
        w_acc = 3.0
        w_footpos = 3.0
        w_delta_acs_nonstand = 3.0
        w_delta_acs_stand = 10.0

        w_array_nonstand = np.array(
            [
                w_mpos,
                w_mvel_nonstand,
                w_ptrans_pos,
                w_ptrans_velx,
                w_ptrans_vely,
                w_prot_pos,
                w_prot_vel,
                w_ptrans_z,
                w_torque,
                w_foot_force,
                w_acc,
                w_footpos,
                w_delta_acs_nonstand,
            ]
        )
        w_array_stand = np.array(
            [
                w_mpos,
                w_mvel_stand,
                w_ptrans_pos,
                w_ptrans_velx,
                w_ptrans_vely,
                w_prot_pos,
                w_prot_vel,
                w_ptrans_z,
                w_torque,
                w_foot_force,
                w_acc,
                w_footpos,
                w_delta_acs_stand,
            ]
        )

        self.reward_weights = w_array_nonstand / np.sum(w_array_nonstand)
        self.reward_weights_stand = w_array_stand / np.sum(w_array_stand)
        self.reward_scales = dict()
        # to balance unit: meter/radians/N/Nm/etc, for reward: exp(-scale*norm)
        self.reward_scales["mpos"] = 5.0
        self.reward_scales["mvel"] = 1e-2
        self.reward_scales["mvel_stand"] = 50.0
        self.reward_scales["prot"] = 10.0
        self.reward_scales["prot_vel"] = 1.0
        self.reward_scales["meter"] = 3.0
        self.reward_scales["meter_z"] = 10.0
        self.reward_scales["meter_velx"] = 2.5
        self.reward_scales["meter_vely"] = 5.0
        self.reward_scales["torque"] = 5e-7
        self.reward_scales["foot_force"] = 2.5e-5
        self.reward_scales["double_foot_force"] = 1.5e-5
        self.reward_scales["side_foot_force"] = 1.5e-3
        self.reward_scales["ppos_acc"] = 2e-5
        self.reward_scales["prot_acc"] = 1e-5
        self.reward_scales["macc"] = 1e-9
        self.reward_scales["foot_pos"] = 100
        self.reward_scales["delta_acs"] = 2.0

        # add a negative sign
        for key in self.reward_scales:
            self.reward_scales[key] *= -1.0

    def __init_step_func(self):
        if self.step_zerotorque:
            self.__sim_step = self.__step_sim_zerotorque
        else:
            self.__sim_step = self.__step_sim_nominal
        self.__get_pTargets = self.acs_norm2actual

    def _init_u(self):
        # p_paras = [400, 200, 200, 500, 20]
        # d_paras = [4, 4, 10, 20, 4]
        # feed forward torque
        torque_fwd = [0, 0, 0, 0, 0]
        self.pd_uncertainty = np.ones((20,))
        # init controller class u
        u = pd_in_t()
        # assign PD controller parameters
        for i in range(5):
            # left leg
            u.leftLeg.motorPd.torque[i] = 0  # torque_fwd[i]  # Feedforward torque
            u.leftLeg.motorPd.pTarget[i] = 0  # p_target[i]
            u.leftLeg.motorPd.pGain[i] = self.pGain[i]
            u.leftLeg.motorPd.dTarget[i] = 0
            u.leftLeg.motorPd.dGain[i] = self.dGain[i]
        u.rightLeg.motorPd = u.leftLeg.motorPd
        return u

    def _update_pd(self):
        self.u.leftLeg.motorPd.pGain[:5] = self.pGain[:5] * self.pd_uncertainty[:5]
        self.u.leftLeg.motorPd.dGain[:5] = self.dGain[:5] * self.pd_uncertainty[10:15]
        self.u.rightLeg.motorPd.pGain[:5] = self.pGain[5:10] * self.pd_uncertainty[5:10]
        self.u.rightLeg.motorPd.dGain[:5] = self.dGain[5:10] * self.pd_uncertainty[15:20]

    def __init_robot_pos(self):
        self.reference_generator.reset()
        (
            ref_base_pos,
            ref_base_rot,
            ref_mpos,
        ) = self.reference_generator.get_init_pose()
        ref_q = euler2quat(ref_base_rot.reshape((3,)))
        ref_base = np.hstack([ref_base_pos.reshape((1, 3)), ref_q.reshape((1, 4))])
        self.set_motor_base_pos(ref_mpos, ref_base)
        self._init_obs_cassie_state(ref_mpos.ravel(), ref_base.ravel())

    def _init_obs_cassie_state(self, ref_mpos, ref_base):
        cassie_out = self.__set_cassie_out(ref_mpos, ref_base)
        self.obs_cassie_state = self.sim.estimate_state(cassie_out)
        self.__add_obs_noise()
        # print("init obs_cassie_state:", np.array(self.obs_cassie_state.pelvis.translationalAcceleration).ravel())

    def __init_action_filter(self):
        self.action_filter = ActionFilterButter(
            lowcut=None,
            highcut=[4],
            sampling_rate=self.real_env_freq,
            order=self.action_filter_order,
            num_joints=self.num_motor,
        )

    def __init_env_randomlizer(self):
        self.env_randomlizer = EnvRandomlizer(self.sim, self.sim_freq)
        self.__set_dynamics_properties()

    def __init_perturbation_generator(self):
        self.pertubration_generator = PerturbationGenerator(self.real_env_freq, 0.2)

    def __reset_action_filter(self):
        self.action_filter.reset()
        noise_rot_magnitute, _, _, _ = self.env_randomlizer.get_rand_nosie()
        default_action = (
            self.qpos[self.motor_idx]
            + np.random.normal(size=self.num_motor) * noise_rot_magnitute
        )
        self.action_filter.init_history(default_action)

    def reset(self):
        self.__reset_env()
        self.action_filter.reset()
        self.pertubration_generator.reset()
        self.env_randomlizer.randomize_dynamics()
        self.__set_dynamics_properties()
        self.__update_data(step=False)
        self.__reset_consts()
        self.__reset_action_filter()
        obs_vf, obs_pol = self.__get_observation(step=False)
        if self.observation_space_vf is None:
            self.__set_obs_space(obs_vf, obs_pol)
        return obs_vf, obs_pol

    def __reset_env(self):
        self.sim.full_reset()
        self.sim.set_time(0)
        self.timestep = 0
        self.time_in_sec = 0.0
        self.previous_obs = deque(maxlen=self.history_len_vf)
        self.previous_acs = deque(maxlen=self.history_len_vf)
        self.long_history = deque(maxlen=self.history_len_pol)

        self.__init_robot_pos()

        self.reward = None
        self.done = None
        self.info = {}
        self.fall_flag = False

    def __reset_consts(self):
        self.init_xy = self.qpos[[0, 1]]
        self.height = self.qpos[2]
        self.foot_pos = np.empty(6)
        self.sim.foot_pos(self.foot_pos)
        self.init_foot_pos = self.foot_pos
        self.applied_force = np.zeros((6,))
        self.last_acs = self.qpos[self.motor_idx]

    def __set_dynamics_properties(self):
        if self.minimal_rand:
            self.sim.set_geom_friction(
                self.env_randomlizer.get_rand_floor_friction(), "floor"
            )
        else:
            self.sim.set_dof_damping(self.env_randomlizer.get_rand_damping())
            self.sim.set_dof_stiffness(self.env_randomlizer.get_rand_stiffness())
            self.sim.set_body_mass(self.env_randomlizer.get_rand_mass())
            self.sim.set_body_ipos(self.env_randomlizer.get_rand_ipos())
            self.sim.set_body_inertia(self.env_randomlizer.get_rand_inertia())
            self.sim.set_geom_friction(
                self.env_randomlizer.get_rand_floor_friction(), "floor"
            )
            self.pd_uncertainty = self.env_randomlizer.get_pd_uncertainty()
            self._update_pd()
            # self.sim.set_geom_quat(self.env_randomlizer.get_floor_slope(), 'floor')

    def __set_obs_space(self, obs_vf, obs_pol):
        self.observation_space_vf = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_vf.shape
        )
        self.observation_space_pol = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_pol[0].shape
        )
        self.observation_space_pol_cnn = obs_pol[1].shape
        # Remove debug print about observation space init

    ##########################################
    #                Step                    #
    ##########################################
    def __step_sim_nominal(self, actual_pTs_filtered):
        """Step the simulation using MPC or PD control."""
        if self.use_mpc:
            try:
                # Detect if we might be in an unstable state
                pelvis_orientation = quat2euler(self.qpos[3:7])
                roll_pitch_tilt = np.abs(pelvis_orientation[:2])  # Roll and pitch
                is_unstable = np.any(roll_pitch_tilt > 0.15)  # ~8.6 degrees
                
                # Get current state
                current_state = self._get_current_state()
                
                # Get reference trajectory
                reference_trajectory = self._generate_mpc_reference_trajectory()
                
                # Compute optimal torques using MPC
                torques = self.mpc_controller.compute_control(current_state, reference_trajectory)
                
                # Apply torques directly - ensure torques is a numpy array
                torques = np.array(torques).flatten()
                
                # Check if torques contains NaN or infinite values
                if np.any(np.isnan(torques)) or np.any(np.isinf(torques)):
                    # Remove debug message, use PD control silently
                    # Fall back to PD control in this step
                    for i in range(5):
                        self.u.leftLeg.motorPd.pTarget[i] = actual_pTs_filtered[i]
                        self.u.rightLeg.motorPd.pTarget[i] = actual_pTs_filtered[i+5]
                        self.u.leftLeg.motorPd.torque[i] = 0  # Zero feedforward torque
                        self.u.rightLeg.motorPd.torque[i] = 0  # Zero feedforward torque
                else:
                    # Limit maximum torque magnitude for safety (prevent extreme movements)
                    max_torque = 50.0  # Further reduced from 100 to 50 for safety
                    torques = np.clip(torques, -max_torque, max_torque)
                
                    # Create temporary arrays for torques to avoid copy attribute issue
                    left_torques = np.zeros(5)
                    right_torques = np.zeros(5)
                    
                    # Copy values individually
                    for i in range(5):
                        left_torques[i] = torques[i]
                        right_torques[i] = torques[i+5]
                    
                    # Use even lower torque scale when stable to minimize disruption
                    # Only use higher MPC correction when potentially unstable
                    if is_unstable:
                        torque_scale = 0.03  # Small when unstable (reduced from 0.05)
                    else:
                        torque_scale = 0.01  # Extremely small when stable to just gently guide
                        
                    # Apply reduced torques with lower gain than before (for stability)
                    for i in range(5):
                        self.u.leftLeg.motorPd.torque[i] = left_torques[i] * torque_scale
                        self.u.rightLeg.motorPd.torque[i] = right_torques[i] * torque_scale
                    
                    # Set position targets from filtered actions for backup PD control
                    for i in range(5):
                        self.u.leftLeg.motorPd.pTarget[i] = actual_pTs_filtered[i]
                        self.u.rightLeg.motorPd.pTarget[i] = actual_pTs_filtered[i+5]
                    
                    # Only reduce PD gains if MPC is producing valid outputs and we're in a stable state
                    if not is_unstable:
                        # Use PD gains mostly at full strength
                        mpc_pd_ratio = 0.9  # Increased from 0.7 to 0.9 to rely even more on PD controller
                        temp_pgain_left = np.zeros(5)
                        temp_pgain_right = np.zeros(5)
                        
                        # Store original PD gains
                        for i in range(5):
                            temp_pgain_left[i] = self.u.leftLeg.motorPd.pGain[i]
                            temp_pgain_right[i] = self.u.rightLeg.motorPd.pGain[i]
                            self.u.leftLeg.motorPd.pGain[i] *= mpc_pd_ratio
                            self.u.rightLeg.motorPd.pGain[i] *= mpc_pd_ratio
                
                # Step the simulation
                for _ in range(self.num_sims_per_env_step):
                    self.obs_cassie_state = self.sim.estimate_state(
                        self.sim.step_pd_without_estimation(self.u)
                    )
                
                # Restore original PD gains if they were modified
                if 'temp_pgain_left' in locals():
                    for i in range(5):
                        self.u.leftLeg.motorPd.pGain[i] = temp_pgain_left[i]
                        self.u.rightLeg.motorPd.pGain[i] = temp_pgain_right[i]
                
            except Exception as e:
                # Remove debug message
                use_mpc_correction = False
                mpc_correction = np.zeros_like(actual_pTs_filtered)
                
                # Fall back to PD control
                for i in range(5):
                    self.u.leftLeg.motorPd.pTarget[i] = actual_pTs_filtered[i]
                    self.u.rightLeg.motorPd.pTarget[i] = actual_pTs_filtered[i+5]
                    self.u.leftLeg.motorPd.torque[i] = 0
                    self.u.rightLeg.motorPd.torque[i] = 0
                    
                for _ in range(self.num_sims_per_env_step):
                    self.obs_cassie_state = self.sim.estimate_state(
                        self.sim.step_pd_without_estimation(self.u)
                    )
        else:
            # Original PD control
            for i in range(5):
                self.u.leftLeg.motorPd.pTarget[i] = actual_pTs_filtered[i]
                self.u.rightLeg.motorPd.pTarget[i] = actual_pTs_filtered[i+5]
                
            for _ in range(self.num_sims_per_env_step):
                self.obs_cassie_state = self.sim.estimate_state(
                    self.sim.step_pd_without_estimation(self.u)
                )

    def __step_sim_zerotorque(self, actual_pTs_filtered):
        # 1 control_step = 0.0005s / 2kHz
        for i in range(5):
            self.u.leftLeg.motorPd.pTarget[i] = 0.0 * actual_pTs_filtered[i]
            self.u.rightLeg.motorPd.pTarget[i] = 0.0 * actual_pTs_filtered[i+5]
            
        for _ in range(self.num_sims_per_env_step):
            # give pTargets to motors
            self.obs_cassie_state = self.sim.estimate_state(
                self.sim.step_pd_without_estimation(self.u)
            )

    def step(self, acs, restore=False):
        if restore:
            self.sim.set_qpos(np.copy(self.qpos_obs))
            self.sim.set_qvel(np.copy(self.qvel_obs))

        # RL Policy outputs normalized action, convert to actual action
        # check joint limit and clip it
        actual_pTs = self.acs_norm2actual(acs)

        if self.use_action_filter:
            actual_pTs_filtered = self.__action_filter(actual_pTs)
        else:
            actual_pTs_filtered = actual_pTs

        # Reference trajectory for stepping
        ref_mpos, ref_base = self._get_reference_trajectory()
        
        # MPC controller integration - new safety-focused approach
        if self.use_mpc:
            # We'll decide whether to use MPC based on stability metrics
            use_mpc_correction = False
            mpc_correction = np.zeros_like(actual_pTs_filtered)
            
            # Check stability metrics
            # 1. Check if pelvis orientation is significantly tilted
            pelvis_orientation = quat2euler(self.qpos[3:7])
            roll_pitch_tilt = np.abs(pelvis_orientation[:2])  # Roll and pitch
            
            # 2. Check angular velocity - high values indicate potential instability
            angular_velocity = self.qvel[3:6]  # Angular velocities
            
            # 3. Check velocity in z direction - negative indicates falling
            z_velocity = self.qvel[2]
            
            # Define thresholds for safety intervention
            roll_pitch_threshold = 0.2  # ~11 degrees in radians
            angular_vel_threshold = 1.0  # rad/s
            z_velocity_threshold = -0.3  # m/s
            
            # Decide whether to use MPC for stabilization
            if (np.any(roll_pitch_tilt > roll_pitch_threshold) or 
                np.any(np.abs(angular_velocity) > angular_vel_threshold) or
                z_velocity < z_velocity_threshold):
                use_mpc_correction = True
                
                # If we're in a critical state, let's compute MPC correction
                try:
                    # Prepare current state for MPC
                    current_state = self._get_current_state()
                    
                    # Generate reference trajectory for MPC horizon
                    reference_trajectory = self._generate_mpc_reference_trajectory()
                    
                    # Compute MPC control action
                    mpc_torques = self.mpc_controller.compute_control(current_state, reference_trajectory)
                    
                    # Check for NaN or inf values in quiet mode
                    if np.any(np.isnan(mpc_torques)) or np.any(np.isinf(mpc_torques)):
                        # Remove debug message
                        mpc_correction = np.zeros_like(actual_pTs_filtered)
                    else:
                        # Store the MPC correction
                        mpc_correction = mpc_torques
                        
                        # Determine how much MPC to blend in based on severity of instability
                        severity = max(
                            np.max(roll_pitch_tilt) / roll_pitch_threshold,
                            np.max(np.abs(angular_velocity)) / angular_vel_threshold,
                            abs(min(z_velocity, 0)) / abs(z_velocity_threshold)
                        )
                        
                        # Cap severity at 1.0 for safety
                        severity = min(severity, 1.0)
                        
                        # Apply MPC correction with adaptive weight based on severity
                        mpc_weight = 0.05 * severity  # Small base correction scaled by severity
                        actual_pTs_filtered = (1 - mpc_weight) * actual_pTs_filtered + mpc_weight * mpc_correction
                except Exception as e:
                    # Remove debug message
                    use_mpc_correction = False
                    mpc_correction = np.zeros_like(actual_pTs_filtered)
        
        if self.step_zerotorque:
            self.__step_sim_zerotorque(actual_pTs_filtered)
        else:
            self.__step_sim_nominal(actual_pTs_filtered)

        # Get reward and next observation
        if not restore:
            self.__update_data(step=True)
        ob_vf, ob_pol = self.__get_observation(acs=actual_pTs_filtered, step=True)
        reward = self.__get_reward(actual_pTs_filtered)
        done = self.__is_done()

        # Check if reward is a tuple (original implementation) or a single value
        reward_sum = 0
        # Info dict for logging
        reward_dict = {}
        
        # Handle different reward implementations
        if isinstance(reward, tuple) and len(reward) > 1:
            reward_sum = np.sum(reward[0])  # First element is total reward
            # Check if reward[1] is a dictionary
            if isinstance(reward[1], dict):
                reward_dict = reward[1]
            else:
                # Iterate through reward components if available
                for i, n in enumerate(self.reward_names):
                    try:
                        # The original code checked if ref_mpos[0] is False to determine if in standing mode
                        # Now we need a different way to check this
                        is_standing_mode = self.reference_generator.in_stand_mode if hasattr(self.reference_generator, 'in_stand_mode') else False
                        
                        if n in ["r_mvel", "r_delta_acs"] and is_standing_mode:
                            reward_dict[n] = 0.0
                        else:
                            if i < len(reward[0]):
                                reward_dict[n] = reward[0][i]
                            else:
                                reward_dict[n] = 0.0
                    except (IndexError, TypeError):
                        reward_dict[n] = 0.0
        else:
            # Handle case where reward is a single value
            reward_sum = reward
            for n in self.reward_names:
                reward_dict[n] = 0.0  # Set all components to 0
                
        info = {"reward_dict": reward_dict}

        return ob_vf, ob_pol, reward_sum, done, info

    def _generate_mpc_reference_trajectory(self):
        """
        Generate reference trajectory for MPC over the prediction horizon.
        Returns a state trajectory of shape (nx, horizon+1)
        """
        if not hasattr(self, 'mpc_controller'):
            raise RuntimeError("MPC controller not initialized but _generate_mpc_reference_trajectory called")
        
        # Number of states and horizon from MPC controller
        nx = self.mpc_controller.nx
        horizon = self.mpc_controller.horizon
        
        # Initialize reference trajectory
        reference_trajectory = np.zeros((nx, horizon + 1))
        
        # Get current reference state
        ref_mpos, ref_base = self._get_reference_trajectory()
        
        # Extract data from ref_base
        ref_base_pos = ref_base[:3]
        
        # If ref_base has 7 values, it's using quaternion representation
        if len(ref_base) == 7:
            # Convert quaternion to Euler angles (simplified)
            ref_base_rot_euler = quat2euler(ref_base[3:7])
        # If it has 6 values, the last 3 are already Euler angles
        elif len(ref_base) == 6:
            ref_base_rot_euler = ref_base[3:6]
        else:
            # Default to zero rotation
            ref_base_rot_euler = np.zeros(3)
        
        # Set reference velocities to zero
        ref_motor_vel = np.zeros(self.num_motor)
        
        # Fill the first reference state
        reference_trajectory[:self.num_motor, 0] = ref_mpos
        reference_trajectory[self.num_motor:2*self.num_motor, 0] = ref_motor_vel
        reference_trajectory[2*self.num_motor:2*self.num_motor+3, 0] = ref_base_pos
        reference_trajectory[2*self.num_motor+3:, 0] = ref_base_rot_euler
        
        # For simplicity, we'll use the same reference for the entire horizon
        # In a more sophisticated implementation, we would use a sequence of reference states
        # that follow a desired trajectory
        for i in range(1, horizon + 1):
            reference_trajectory[:, i] = reference_trajectory[:, 0]
        
        return reference_trajectory
        
    def _get_current_state(self):
        """
        Get the current state vector for the MPC controller.
        Returns a vector [motor_pos, motor_vel, base_pos, base_rot]
        """
        if not hasattr(self, 'mpc_controller'):
            raise RuntimeError("MPC controller not initialized but _get_current_state called")
            
        # Get current motor positions and velocities
        motor_pos = self.qpos[self.motor_idx]
        motor_vel = self.qvel[self.motor_vel_idx]
        
        # Get base position and orientation
        base_pos = self.qpos[self.base_idx[0:3]]
        
        # Get rotation as Euler angles 
        # If using quaternion (length 4), convert to Euler angles
        if len(self.qpos) >= 7 and self.base_idx[3] == 3:  # Assuming base_idx[3:7] points to quaternion
            base_rot_euler = quat2euler(self.qpos[3:7])
        else:
            # Otherwise directly use Euler angles
            base_rot_euler = self.qpos[self.base_idx[3:6]]
        
        # Check expected state dimensions from MPC controller
        expected_dim = self.mpc_controller.nx
        current_dim = len(motor_pos) + len(motor_vel) + len(base_pos) + len(base_rot_euler)
        
        # Combine into a single state vector
        current_state = np.concatenate([motor_pos, motor_vel, base_pos, base_rot_euler])
        
        # Print warning if dimensions don't match
        if current_dim != expected_dim:
            print(f"Warning: State dimension mismatch in _get_current_state. Returning {current_dim} dims, but MPC expects {expected_dim}.")
        
        return current_state

    def __update_data(self, step=True):
        self.qpos = np.copy(self.sim.qpos())
        self.qvel = np.copy(self.sim.qvel())
        self.qacc = np.copy(self.sim.qacc())
        self.height = self.qpos[2]
        self.foot_pos = np.empty(6)
        self.sim.foot_pos(self.foot_pos)
        self.curr_rpy_gt = quat2euler(self.qpos[3:7])
        self.curr_rpy_obs = quat2euler(self.obs_cassie_state.pelvis.orientation)
        if step:
            self.timestep += 1
            self.time_in_sec = (
                self.timestep * self.num_sims_per_env_step
            ) / self.sim_freq
            self.reference_generator.update_ref_env(
                self.time_in_sec, self.qpos[:2], self.curr_rpy_gt[-1]
            )
        self.ref_dict = self.reference_generator.get_ref_motion()

    def __add_obs_noise(self):
        noise_vec = np.random.normal(size=(32,))  # 20+3+3+3+3
        (
            noise_rot,
            noise_rot_vel,
            noise_linear_acc,
            noise_linear_vel,
        ) = self.env_randomlizer.get_rand_nosie()
        obs_euler = quat2euler(self.obs_cassie_state.pelvis.orientation)
        self.obs_cassie_state.pelvis.translationalVelocity[:3] += (
            noise_vec[:3] * noise_linear_vel
        )
        self.obs_cassie_state.pelvis.translationalAcceleration[:3] += (
            noise_vec[3:6] * noise_linear_acc
        )
        self.obs_cassie_state.pelvis.rotationalVelocity[:3] += (
            noise_vec[6:9] * noise_rot_vel
        )
        obs_euler[:3] += noise_vec[9:12] * noise_rot
        obs_quat = euler2quat(obs_euler)
        self.obs_cassie_state.pelvis.orientation[:4] = obs_quat[:4]
        self.obs_cassie_state.motor.position[:10] += noise_vec[12:22] * noise_rot
        self.obs_cassie_state.motor.velocity[:10] += noise_vec[22:32] * noise_rot_vel

    def __set_cassie_out(self, mpos, base_pos):
        cassie_out = cassie_out_t()
        cassie_out.pelvis.vectorNav.orientation[:4] = base_pos[3:7]
        # motor, leftleg
        cassie_out.leftLeg.hipRollDrive.position = mpos[0]
        cassie_out.leftLeg.hipYawDrive.position = mpos[1]
        cassie_out.leftLeg.hipPitchDrive.position = mpos[2]
        cassie_out.leftLeg.kneeDrive.position = mpos[3]
        cassie_out.leftLeg.footDrive.position = mpos[4]
        # motor, rightleg
        cassie_out.rightLeg.hipRollDrive.position = mpos[5]
        cassie_out.rightLeg.hipYawDrive.position = mpos[6]
        cassie_out.rightLeg.hipPitchDrive.position = mpos[7]
        cassie_out.rightLeg.kneeDrive.position = mpos[8]
        cassie_out.rightLeg.footDrive.position = mpos[9]
        return cassie_out

    def __apply_perturbation(self):
        (
            force_to_apply,
            apply_force_flag,
        ) = self.pertubration_generator.apply_perturbation()
        if apply_force_flag:
            self.applied_force = force_to_apply
            self.sim.apply_force(self.applied_force)
            if self.is_visual:
                print("Applied Perturbation: ", self.applied_force)
        else:
            self.applied_force *= 0.0
            self.sim.clear_forces()

    ##########################################
    #              Observation               #
    ##########################################
    def __get_observation(self, acs=np.zeros(10), step=False):
        ref_dict_1 = self.reference_generator.get_ref_motion(look_forward=1)
        ref_dict_4 = self.reference_generator.get_ref_motion(look_forward=4)
        ref_dict_7 = self.reference_generator.get_ref_motion(look_forward=7)

        ob1 = ref_dict_1["motor_pos"]
        ob4 = ref_dict_4["motor_pos"]
        ob7 = ref_dict_7["motor_pos"]

        ob_curr = np.concatenate(
            [
                np.array(self.obs_cassie_state.pelvis.translationalVelocity).ravel(),
                np.array(self.obs_cassie_state.pelvis.orientation).ravel(),
                np.array(self.obs_cassie_state.motor.position).ravel(),
                np.array(self.obs_cassie_state.motor.velocity).ravel(),
            ]
        )

        # ground truth observation
        curr_xy_gt = self.qpos[[0, 1]] - self.init_xy
        ob_curr_gt = np.concatenate(
            [
                self.qvel[[0, 1, 2]],
                self.qacc[[0, 1, 2]],
                self.qpos[3:7],
                self.qpos[self.motor_idx],
                self.qvel[self.motor_vel_idx],
            ]
        )

        # command
        ref_yaw = self.ref_dict["base_rot_global"][-1]
        ob_command = np.concatenate(
            [
                [self.ref_dict["base_pos_global"][-1]],  # walking height
                self.ref_dict["base_vel_local"][[0, 1]],  # local velocity command
                [math.cos(ref_yaw), math.sin(ref_yaw)],  # desired turning yaw angle
            ]
        )  #  height, vx, vy, yaw
        
        # Include MPC suggestions in observation if MPC is enabled
        mpc_suggestion = np.zeros(self.num_motor)
        mpc_stability_metrics = np.zeros(3)  # roll tilt, pitch tilt, z velocity
        
        if self.use_mpc:
            try:
                # Get stability metrics
                pelvis_orientation = quat2euler(self.qpos[3:7])
                roll_pitch_tilt = np.abs(pelvis_orientation[:2])  # Roll and pitch 
                z_velocity = self.qvel[2]
                mpc_stability_metrics = np.array([roll_pitch_tilt[0], roll_pitch_tilt[1], z_velocity])
                
                # Generate MPC suggestion for policy to consider
                current_state = self._get_current_state()
                reference_trajectory = self._generate_mpc_reference_trajectory()
                mpc_suggestion = self.mpc_controller.compute_control(current_state, reference_trajectory)
                
                # Handle potential NaN or inf values
                if np.any(np.isnan(mpc_suggestion)) or np.any(np.isinf(mpc_suggestion)):
                    mpc_suggestion = np.zeros(self.num_motor)
            except Exception as e:
                # Remove debug message
                mpc_suggestion = np.zeros(self.num_motor)
                
        if self.timestep == 0:
            [self.previous_obs.append(ob_curr) for i in range(self.history_len_vf)]
            [
                self.previous_acs.append(np.zeros(self.num_motor))
                for _ in range(self.history_len_vf)
            ]
            [
                self.long_history.append(
                    np.concatenate([ob_curr, np.zeros(self.num_motor)])
                )
                for _ in range(self.history_len_pol)
            ]
        ob_prev = np.concatenate(
            [np.array(self.previous_obs).ravel(), np.array(self.previous_acs).ravel()]
        )
        obs_pol_hist = np.flip(np.asarray(self.long_history).T, 1)

        # Include MPC suggestion in policy observation
        obs_pol_base = np.concatenate([
            ob_prev, 
            ob_curr, 
            ob1, 
            ob4, 
            ob7, 
            ob_command, 
            mpc_suggestion,  # Include MPC action suggestion
            mpc_stability_metrics  # Include stability metrics
        ])
        obs_pol = (obs_pol_base, obs_pol_hist)

        # Also include MPC information in value function observation
        obs_vf = np.concatenate(
            [
                ob_prev,
                ob_curr_gt,
                ob1,
                ob4,
                ob7,
                ob_command,
                self.ref_dict["base_pos_global"][[0, 1]] - curr_xy_gt,
                np.array([self.qpos[2]]),
                self.sim.get_foot_forces(),
                self.env_randomlizer.get_rand_floor_friction(),
                mpc_suggestion,  # Include MPC suggestion
                mpc_stability_metrics  # Include stability metrics
            ]
        )

        if step:
            self.previous_obs.append(ob_curr)
            self.previous_acs.append(acs)
            self.long_history.append(np.concatenate([ob_curr, acs]))
        return obs_vf, obs_pol

    ##########################################
    #                Reward                  #
    ##########################################
    def __get_reward(self, acs):
        # NOTE: reward is using qpos/qvel that don't have noise and delay
        mpos_err = np.sum(
            np.square(self.ref_dict["motor_pos"] - self.qpos[self.motor_idx])
        )
        r_mpos = np.exp(self.reward_scales["mpos"] * mpos_err)

        mvel_err = np.sum(
            np.square(self.ref_dict["motor_vel"] - self.qvel[self.motor_vel_idx])
        )
        if not self.reference_generator.in_stand_mode:
            r_mvel = np.exp(self.reward_scales["mvel"] * mvel_err)
        else:
            r_mvel = np.exp(self.reward_scales["mvel_stand"] * mvel_err)

        curr_xy = self.qpos[[0, 1]] - self.init_xy
        base_pos_err = np.sum(
            np.square(self.ref_dict["base_pos_global"][[0, 1]] - curr_xy)
        )  # pos global
        r_ptrans = np.exp(self.reward_scales["meter"] * base_pos_err)

        velocity_local = global2local(self.qvel[:3], self.curr_rpy_gt)

        r_ptrans_velx = np.exp(
            self.reward_scales["meter_velx"]
            * abs(self.ref_dict["base_vel_local"][0] - velocity_local[0])
        )
        # print(
        #     "ptrans_velx reward: {:.6f}, desired: {:.3f}, actual: {:.3f}".format(
        #         r_ptrans_velx, self.ref_dict["base_vel_local"][0], velocity_local[0]
        #     ),
        #     end="\r",
        # )

        r_ptrans_vely = np.exp(
            self.reward_scales["meter_vely"]
            * abs(self.ref_dict["base_vel_local"][1] - velocity_local[1])
        )
        # print(
        #     "ptrans_vely reward: {:.6f}, desired: {:.3f}, actual: {:.3f}".format(
        #         r_ptrans_vely, self.ref_dict["base_vel_local"][1], velocity_local[1]
        #     ),
        #     end="\r",
        # )

        base_angle_err = np.sum(
            1.0 - np.cos(self.ref_dict["base_rot_global"] - self.curr_rpy_gt)
        )
        r_prot = np.exp(self.reward_scales["prot"] * base_angle_err)
        # print(
        #     "prot reward:{:.6f}, desired: {:.4f}, actual: {:.4f}".format(
        #         r_prot,
        #         np.rad2deg(self.ref_dict["base_rot_global"][-1]),
        #         np.rad2deg(self.curr_rpy_gt[-1]),
        #     ),
        #     end="\r",
        # )

        base_anglevel_err = np.sum(
            np.square(
                self.qvel[[3, 4, 5]]
                - np.array([0.0, 0.0, self.ref_dict["base_vel_local"][-1]])
            )
        )  # stablize roll and pitch and track yaw vel
        r_prot_vel = np.exp(self.reward_scales["prot_vel"] * base_anglevel_err)

        ptrans_z_err = np.square(self.qpos[2] - self.ref_dict["base_pos_global"][2])
        r_ptrans_z = np.exp(self.reward_scales["meter_z"] * ptrans_z_err)

        torques_err = np.sum(np.square(self._calc_torque(acs)))
        r_torques = np.exp(self.reward_scales["torque"] * torques_err)

        impact_forces = self.sim.get_foot_impact_forces()
        if self.reference_generator.in_stand_mode:
            foot_forces = self.sim.get_foot_forces()
            foot_forces_side = foot_forces[[0, 1, 3, 4]]
            if any(abs(impact_forces) <= 1.0):
                r_foot_force = 0.0
            else:
                foot_force_err = np.sum(np.square(foot_forces_side))
                r_foot_force = np.exp(
                    self.reward_scales["side_foot_force"] * foot_force_err
                )
        else:
            if abs(impact_forces[0]) <= 0.1 and abs(impact_forces[1]) <= 0.1:
                r_foot_force = 0.0
            elif abs(impact_forces[0]) >= 0.1 and abs(impact_forces[1]) >= 0.1:
                foot_force_err = np.square(np.sum(impact_forces) / 2.0)
                r_foot_force = np.exp(
                    self.reward_scales["double_foot_force"] * foot_force_err
                )
            else:
                foot_force_err = np.sum(np.square(impact_forces))
                r_foot_force = np.exp(self.reward_scales["foot_force"] * foot_force_err)

        ref_footpos = self.get_foot_pos_absolute(
            self.ref_dict["base_pos_global"],
            self.ref_dict["base_rot_global"],
            self.ref_dict["motor_pos"],
        )
        foot_pos_err = np.sum(
            np.square(
                ref_footpos[[2, 5]]
                - self.offset_footJoint2midFoot
                - self.foot_pos[[2, 5]]
            )
        )
        r_footpos = np.exp(self.reward_scales["foot_pos"] * foot_pos_err)

        base_xyz_acc_err = np.sum(np.square(self.qacc[0:3]))
        base_rxyz_acc_err = np.sum(np.square(self.qacc[3:6]))
        base_motor_acc_err = np.sum(np.square(self.qacc[self.motor_vel_idx]))
        r_base_xyz_acc = np.exp(self.reward_scales["ppos_acc"] * base_xyz_acc_err)
        r_base_rxyz_acc = np.exp(self.reward_scales["prot_acc"] * base_rxyz_acc_err)
        r_motor_acc = np.exp(self.reward_scales["macc"] * base_motor_acc_err)
        r_acc = (r_base_xyz_acc + r_base_rxyz_acc + r_motor_acc) / 3.0

        r_delta_acs = np.exp(
            self.reward_scales["delta_acs"] * np.sum(np.square(self.last_acs - acs))
        )

        # NOTE: should be in the same order with self.reward_weights
        rewards = np.array(
            [
                r_mpos,
                r_mvel,
                r_ptrans,
                r_ptrans_velx,
                r_ptrans_vely,
                r_prot,
                r_prot_vel,
                r_ptrans_z,
                r_torques,
                r_foot_force,
                r_acc,
                r_footpos,
                r_delta_acs,
            ]
        )

        if not self.reference_generator.in_stand_mode:
            total_reward = np.sum(self.reward_weights * rewards)
            reward_dict = dict(zip(self.reward_names, self.reward_weights * rewards))
        else:
            total_reward = np.sum(self.reward_weights_stand * rewards)
            reward_dict = dict(
                zip(self.reward_names, self.reward_weights_stand * rewards)
            )
        return total_reward, reward_dict

    ##########################################
    #           Early Termination            #
    ##########################################
    def __is_done(self):
        tarsus_pos = self.get_tarsus_pos(
            self.qpos[[0, 1, 2]], self.qpos[4:7], self.qpos[self.motor_idx]
        )
        if self.height < FALLING_THRESHOLD:
            self.fall_flag = True
            return True
        elif any(tarsus_pos[[2, 5]] <= TARSUS_HITGROUND_THRESHOLD):
            self.fall_flag = True
            return True
        elif self.timestep >= self.max_timesteps:
            return True
        # Additional fall detection checks
        elif self.height < 0.6:  # If robot is too close to the ground but above FALLING_THRESHOLD
            self.fall_flag = True
            return True
        # Check for extreme tilt angles
        elif abs(self.curr_rpy_gt[0]) > 0.7 or abs(self.curr_rpy_gt[1]) > 0.7:  # About 40 degrees
            self.fall_flag = True
            return True
        # Check for high motor velocities (potential instability)
        elif np.any(np.abs(self.qvel[self.motor_vel_idx]) > 30):  # Threshold for motor velocities
            self.fall_flag = True
            return True
        else:
            return False

    ##########################################
    #                 Utils                  #
    ##########################################
    def acs_actual2norm(self, actual_acs):
        return (actual_acs - self.safe_action_bounds[0]) * 2 / (
            self.safe_action_bounds[1] - self.safe_action_bounds[0]
        ) - 1

    def acs_norm2actual(self, acs):
        return self.safe_action_bounds[0] + (acs + 1) / 2.0 * (
            self.safe_action_bounds[1] - self.safe_action_bounds[0]
        )

    def set_motor_base_pos(self, motor_pos, base_pos, iters=1500):
        """
        Kind of hackish.
        This takes a floating base position and some joint positions
        and abuses the MuJoCo solver to get the constrained forward
        kinematics.
        There might be a better way to do this, e.g. using mj_kinematics
        """
        for _ in range(iters):
            qpos = np.copy(self.sim.qpos())
            qvel = np.copy(self.sim.qvel())

            qpos[self.motor_idx] = motor_pos
            qpos[self.base_idx] = base_pos

            self.sim.set_qpos(qpos)
            self.sim.set_qvel(0 * qvel)

            self.sim.step_pd_without_estimation(pd_in_t())
            self.obs_cassie_state = self.sim.estimate_state(cassie_out_t())
            self.sim.set_time(self.time_in_sec)

    def close(self):
        if self.is_visual:
            self.vis.__del__()
        self.sim.__del__()

    def render(self):
        return self.vis.draw(self.sim)

    def _calc_torque(self, actual_pTs):
        torques = self.pGain * (
            actual_pTs - np.array(self.obs_cassie_state.motor.position)
        ) - self.dGain * np.array(self.obs_cassie_state.motor.velocity)
        return torques

    def get_foot_pos_relative(self, base_rot, motor_pos):
        return self.cassie_fk.get_foot_pos(motor_pos, [0, 0, 0], base_rot)

    def get_foot_pos_absolute(self, base_pos, base_rot, motor_pos):
        return self.cassie_fk.get_foot_pos(motor_pos, base_pos, base_rot)

    def get_tarsus_pos(self, base_pos, base_rot, motor_pos):
        return self.cassie_fk.get_tarsus_pos(motor_pos, base_pos, base_rot)

    def _get_reference_trajectory(self):
        """
        Get the current reference trajectory for the MPC controller.
        Returns a tuple (ref_mpos, ref_base) 
        """
        if not hasattr(self, 'reference_generator'):
            raise RuntimeError("Reference generator not initialized but _get_reference_trajectory called")
        
        # The get_ref_motion method returns a dictionary, not a tuple
        ref_dict = self.reference_generator.get_ref_motion(look_forward=0)
        
        # Extract the relevant information from the dictionary
        # Check if ref_dict contains motor positions, otherwise use a default
        if "motor_pos" in ref_dict:
            ref_mpos = ref_dict["motor_pos"]
        else:
            ref_mpos = np.zeros(self.num_motor)
            
        # Check if ref_dict contains base positions and rotations, otherwise use defaults
        if "base_pos_global" in ref_dict and "base_rot_global" in ref_dict:
            ref_base_pos = ref_dict["base_pos_global"]
            ref_base_rot = ref_dict["base_rot_global"]
            ref_base = np.concatenate([ref_base_pos, ref_base_rot])
        else:
            # Default standing pose
            ref_base = np.zeros(7)  # 3 for position, 4 for quaternion rotation
            ref_base[2] = 1.0  # Default z-height
            ref_base[3:7] = [1, 0, 0, 0]  # Default quaternion (no rotation)
            
        return ref_mpos, ref_base

    def __action_filter(self, action):
        """Apply the Butterworth filter to the input action."""
        if self.use_action_filter:
            return self.action_filter.filter(action)
        else:
            return action
