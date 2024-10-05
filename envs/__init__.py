from gymnasium.envs.registration import register


register(
    id="Pid-v0",
    entry_point="envs.pid:VSSPIDTuningEnv",
    kwargs={"max_steps": 1200, "repeat_action": 1},
    max_episode_steps=1200,
)

register(
    id="Pid-v1",
    entry_point="envs.pid:VSSPIDTuningEnv",
    kwargs={"max_steps": 75, "repeat_action": 16},
    max_episode_steps=75,
)

register(
    id="Penalty-v0",
    entry_point="envs.penalty:VSSPenaltyEnv",
    kwargs={"max_steps": 1200, "repeat_action": 1},
    max_episode_steps=1200,
)

register(
    id="Penalty-v1",
    entry_point="envs.penalty:VSSPenaltyEnv",
    kwargs={"max_steps": 75, "repeat_action": 16},
    max_episode_steps=75,
)

register(
    id="Attacker-v0",
    entry_point="envs.attacker:VSSAttackerEnv",
    kwargs={"max_steps": 1200, "repeat_action": 1},
    max_episode_steps=1200,
)

register(
    id="Attacker-v1",
    entry_point="envs.attacker:VSSAttackerEnv",
    kwargs={"max_steps": 75, "repeat_action": 16},
    max_episode_steps=75,
)

register(
    id="Attacker-v2",
    entry_point="envs.vssef:VSSEF"
)

register(
    id="GoTo-v0",
    entry_point="envs.goto:VSSGoToEnv",
    kwargs={"max_steps": 120},
    max_episode_steps=120,
)


# register(
#     id="Enhanced-v0",
#     entry_point="envs.enhanced:SSLPathPlanningEnv",
#     kwargs={"field_type": 2, "n_robots_yellow": 0, "repeat_action": 1},
#     max_episode_steps=1200,
# )

# register(
#     id="Enhanced-v1",
#     entry_point="envs.enhanced:SSLPathPlanningEnv",
#     kwargs={"field_type": 2, "n_robots_yellow": 0, "repeat_action": 16},
#     max_episode_steps=75,
# )

# register(
#     id="Obstacle-v0",
#     entry_point="envs.obstacles:ObstacleEnv",
#     kwargs={"n_robots_yellow": 1, "repeat_action": 1},
#     max_episode_steps=1200,
# )

# register(
#     id="Obstacle-v1",
#     entry_point="envs.obstacles:ObstacleEnv",
#     kwargs={"n_robots_yellow": 1, "repeat_action": 16},
#     max_episode_steps=75,
# )

# register(
#     id="Test-v0",
#     entry_point="envs.enhanced:TestEnv",
#     max_episode_steps=1200,
# )

# register(
#     id="Test-v1",
#     entry_point="envs.enhanced:TestEnv",
#     kwargs={"field_type": 2, "n_robots_yellow": 0, "repeat_action": 16},
#     max_episode_steps=75,
# )

# register(
#     id="TestObstacle-v0",
#     entry_point="envs.obstacles:TestObstacleEnv",
#     max_episode_steps=1200,
# )
