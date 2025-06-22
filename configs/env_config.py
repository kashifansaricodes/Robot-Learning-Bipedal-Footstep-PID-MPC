# config for test, play.py
config_play = {
    "max_timesteps": 2500,  # NOTE: max timestep per episode
    "is_visual": True,  # NOTE: set true to visualize robot
    "minimal_rand": False,  # NOTE: set true to use minimal rand range -> only rand floor friction
    "is_noisy": True,  # NOTE: set true to add noise on observation
    "add_standing": True,  # NOTE: set true to include standing skill
    "fixed_gait": True,  # NOTE: set true to only test one cmd of 'fixed_gait_cmd'
    "add_rotation": False,  # NOTE: set true to add rotation (vyaw) command, only useful if not fixed gait
    "fixed_gait_cmd": [
        1.0,  # vx, m/s
        0.0,  # vy, m/s
        0.98,  # walking height, m
        0.0,  # turning rate, deg/s
    ],  # fixed gait cmd
}

# config for trainig, train.py
config_train = {
    "max_timesteps": 2500,  # NOTE: max timestep per episode
    "is_visual": False,  # NOTE: set true to visualize robot
    "minimal_rand": False,  # NOTE: set true to use minimal rand range -> only rand floor friction
    "is_noisy": False,  # NOTE: set true to add noise on observation
    "add_standing": True,  # NOTE: set true to include standing skill
    "fixed_gait": True,  # NOTE: set true to only test one cmd of 'fixed_gait_cmd'
    "add_rotation": True,  # NOTE: set true to add rotation (vyaw) command, only useful if not fixed gait
    "fixed_gait_cmd": [
        0.0,
        0.0,
        0.98,
        0.0,
    ],  # fixed gait cmd: vx, vy, walking height, vyaw
    "swing_duration": 0.25,
    "stance_duration": 0.25,
    "total_duration": 0.5,
    "init_motor_pos": [
        0.0041,   # left-hip-roll
        0.9163,   # left-hip-yaw
        0.4386,   # left-hip-pitch
        -1.12,    # left-knee
        0.6975,   # left-foot
        -0.0041,  # right-hip-roll
        -0.9163,  # right-hip-yaw
        0.4386,   # right-hip-pitch
        -1.12,    # right-knee
        0.6975,   # right-foot
    ],
    "init_base_pos": [0.0, 0.0, 1.0],
    "init_base_rot": [0.0, 0.0, 0.0, 1.0],
    "use_footstep_plan": True,
}