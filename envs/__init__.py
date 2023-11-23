from gymnasium.envs.registration import register

register(
    id="Baseline-v0",
    entry_point="envs.baseline:SSLPathPlanningBaseLineEnv",
    kwargs={"field_type": 2, "n_robots_yellow": 0},
    max_episode_steps=1200,
)

register(
    id="Enhanced-v0",
    entry_point="envs.enhanced:SSLPathPlanningEnv",
    kwargs={"field_type": 2, "n_robots_yellow": 0, "repeat_action": 1},
    max_episode_steps=1200,
)

register(
    id="Enhanced-v1",
    entry_point="envs.enhanced:SSLPathPlanningEnv",
    kwargs={"field_type": 2, "n_robots_yellow": 0, "repeat_action": 16},
    max_episode_steps=75,
)

register(
    id="Obstacle-v0",
    entry_point="envs.obstacles:SSLPathPlanningMediumEnv",
    kwargs={"n_robots_yellow": 1},
    max_episode_steps=75,
)

register(
    id="Trajectory-v0",
    entry_point="envs.trajectory:TrajectoryEnv",
    max_episode_steps=1,
)