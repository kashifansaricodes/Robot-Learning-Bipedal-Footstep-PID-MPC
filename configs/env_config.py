# config for test, play.py
config_play = {
    "max_timesteps": 8000,
    "is_visual": True,
    "cam_track_robot": True,
    "step_zerotorque": False,
    "minimal_rand": True,
    "is_noisy": False,
    "add_perturbation": False,
    "add_standing": True,
    "fixed_gait": True,
    "add_rotation": False,
    "fixed_gait_cmd": [
        1.0,  # vx, m/s
        0.0,  # vy, m/s
        0.98,  # walking height, m
        0.0,  # turning rate, deg/s
    ],
    "use_mpc": False,
    "use_action_filter": True,
}

# config for trainig, train.py
config_train = {
    "max_timesteps": 8000,
    "is_visual": False,
    "cam_track_robot": False,
    "step_zerotorque": False,
    "minimal_rand": False,
    "is_noisy": True,
    "add_perturbation": True,
    "add_standing": True,
    "fixed_gait": True,
    "add_rotation": True,
    "fixed_gait_cmd": [
        0.0,
        0.0,
        0.98,
        0.0,
    ],
    "use_mpc": False,
    "use_action_filter": True,
}

# config for gaitlibary_test.py
config_static = {
    "max_timesteps": 2500,  # NOTE: max timestep per episode
    "is_visual": True,  # NOTE: set true to visualize robot
    "cam_track_robot": False,  # NOTE: set true to let camera track the robot
    "step_zerotorque": True,  # NOTE: set true to step without torque
    "minimal_rand": True,  # NOTE: set true to use minimal rand range -> only rand floor friction
    "is_noisy": False,  # NOTE: set true to add noise on observation
    "add_perturbation": False,  # NOTE: set true to add external perturbation
    "add_standing": False,  # NOTE: set true to include standing skill
    "fixed_gait": True,  # NOTE: set true to only test one cmd of 'fixed_gait_cmd'
    "add_rotation": False,  # NOTE: set true to add rotation (vyaw) command, only useful if not fixed gait
    "fixed_gait_cmd": [
        0.0,
        0.0,
        0.98,
        0.0,
    ],  # fixed gait cmd: vx, vy, walking height, vyaw
    "use_action_filter": True,
}

# MPC configuration
config_mpc = {
    # Environment parameters
    "max_timesteps": 1000,  # Maximum timesteps per episode
    "control_dt": 0.03,  # Control timestep
    "sim_dt": 0.001,  # Simulation timestep
    "num_sims_per_env_step": 30,  # Number of simulation steps per environment step
    
    # Visualization
    "is_visual": True,  # Whether to visualize the environment
    "cam_track_robot": True,  # Whether to track the robot with the camera
    
    # Randomization
    "randomize_dynamics": False,  # Whether to randomize dynamics
    "randomize_clock": False,  # Whether to randomize clock
    "minimal_rand": True,  # Minimal randomization
    "is_noisy": False,  # Whether to add noise to observations
    
    # Perturbation
    "add_perturbation": False,  # Whether to apply perturbations
    "perturb_time": 0.3,  # Duration of perturbation
    "perturb_magnitude": 100.0,  # Magnitude of perturbation
    
    # Gait parameters
    "add_standing": False,  # Whether to include standing skill
    "fixed_gait": False,  # Whether to use fixed gait
    "add_rotation": True,  # Whether to add rotation commands
    "fixed_gait_cmd": [0.0, 0.0],  # Fixed gait command if using fixed gait
    
    # Simulation options
    "step_zerotorque": False,  # Whether to step with zero torque
    
    # MPC parameters
    "use_mpc": True,  # Whether to use MPC
    "mpc_horizon": 10,  # MPC prediction horizon
    "mpc_dt": 0.01,  # MPC timestep
    
    # Action filter
    "use_action_filter": True,  # Whether to use action filter
    
    # Reference generator parameters
    "ref_generator": {
        "start_standing": False,  # Whether to start in standing mode
        "time_standing_start": 0.0,  # Time to start standing
        "time_standing_duration": 0.0,  # Duration of standing
    },
    
    # Action filter parameters
    "action_filter_order": 2,  # Order of action filter
    "real_env_freq": 33.33,  # Real environment frequency
}

# MPC Configurations
config_mpc_play = {
    "is_visual": True,
    "cam_track_robot": True,
    "max_timesteps": 8000,
    "minimal_rand": True,
    "is_noisy": False,
    "add_perturbation": False,
    "step_zerotorque": False,
    "use_mpc": True,
    "mpc_horizon": 10,
    "mpc_dt": 0.01,
    "add_standing": True,
    "fixed_gait": True,
    "add_rotation": False,
    "fixed_gait_cmd": [
        1.0,  # vx, m/s
        0.0,  # vy, m/s
        0.98,  # walking height, m
        0.0,  # turning rate, deg/s
    ],
    "use_action_filter": True,
}

config_mpc_train = {
    "is_visual": False,
    "cam_track_robot": False,
    "max_timesteps": 8000,
    "minimal_rand": False,
    "is_noisy": True,
    "add_perturbation": True,
    "step_zerotorque": False,
    "use_mpc": True,
    "mpc_horizon": 10,
    "mpc_dt": 0.01,
    "add_standing": True,
    "fixed_gait": True,
    "add_rotation": True,
    "fixed_gait_cmd": [
        0.0,  # vx, m/s
        0.0,  # vy, m/s
        0.98,  # walking height, m
        0.0,  # turning rate, deg/s
    ],
    "use_action_filter": True,
}
