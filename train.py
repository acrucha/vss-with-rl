# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import gymnasium as gym
import numpy as np
import wandb

import envs

from pyvirtualdisplay import Display

from methods.sac import SAC
from utils.experiment import get_experiment, make_env
from utils.experiment import parse_args
from utils.experiment import setup_run
from utils.experiment import get_images


def train(args, exp_name, wandb_run, artifact):
    # environments = gym.vector.AsyncVectorEnv(
    #     [make_env(args, i, exp_name) for i in range(args.num_envs)]
    # )

    environment = make_env(args, 0, exp_name)

    agent = SAC(
        args, environment.observation_space, environment.action_space
    )

    start_time = time.time()
    obs, _ = environment.reset()

    log = {}
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            action = environment.action_space.sample()
        else:
            action = agent.get_action(obs)

        next_obs, reward, done, trunc, info = environment.step(action)

        if "final_info" in info:
            for info in info["final_info"]:
                if info:
                    print(
                        f"global_step={global_step}, episodic_return={info['reward_total']}"
                    )
                    keys_to_log = [x for x in info.keys() if x.startswith("reward_")]
                    for key in keys_to_log:
                        log[f"ep_info/{key.replace('reward_', '')}"] = info[key]
                    break

        if done or trunc:
            next_obs, _ = environment.reset()
            print("Environments reset")

        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            update_actor = global_step % args.policy_frequency == 0
            policy_loss, qf1_loss, qf2_loss, alpha_loss = agent.update(
                args.batch_size, update_actor
            )

            if global_step % args.target_network_frequency == 0:
                agent.critic_target.sync(args.tau)

            if global_step % 100 == 0:
                log.update(
                    {
                        "losses/Value1_loss": qf1_loss.item(),
                        "losses/Value2_loss": qf2_loss.item(),
                        "losses/alpha": agent.alpha,
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                    }
                )

                if update_actor:
                    log.update({"losses/policy_loss": policy_loss.item()})
                # if args.autotune:
                #     log.update({"losses/alpha_loss": alpha_loss.item()})

        wandb.log(log, global_step)
        if global_step % 9999 == 0:
            agent.save(f"models/{exp_name}/")

    artifact.add_file(f"models/{exp_name}/actor.pt")
    wandb_run.log_artifact(artifact)
    environment.close()


def main(params):
    exp_name = f"{params.env}-{params.setup}_{int(time.time())}"
    _display = Display(visible=0, size=(1400, 900))
    _display.start()
    wandb_run, artifact = setup_run(exp_name, params, params.project)
    train(params, exp_name, wandb_run, artifact)


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
