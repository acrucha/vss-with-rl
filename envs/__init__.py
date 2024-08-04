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
    id="Vss-Vision-v0",
    entry_point="envs.vss-vision:VSSVisionEnv",
    kwargs={"max_steps": 1200, "repeat_action": 1},
    max_episode_steps=1200,
)

register(
    id="Vss-Vision-v1",
    entry_point="envs.vss-vision:VSSVisionEnv",
    kwargs={"max_steps": 75, "repeat_action": 16},
    max_episode_steps=75,
)