import gymnasium as gym
from yaml import safe_load
from utils.experiment import base_hyperparams
import sys

if __name__ == "__main__":

    env_name = sys.argv[1] if len(sys.argv) > 1 else "Goto-v0"

    with open("experiments.yml", "r") as f:
        params = safe_load(f)
    experiment = base_hyperparams()

    # Using VSS Single Agent env
    env = gym.make(env_name, render_mode="human")

    env.reset()
    # Run for 1 episode and print reward at the end
    for i in range(1):
        terminated = False
        truncated = False
        while not (terminated or truncated):
            # Step using random actions
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
        print(reward)