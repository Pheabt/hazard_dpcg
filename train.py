import argparse
import os
import random
import sys
import time
import yaml

import numpy as np
import torch

from baselines import logger

from dcpg.algos import *
from dcpg.envs import make_envs
from dcpg.models import *
from dcpg.sample_utils import sample_episodes
from dcpg.storages import RolloutStorage
from test import evaluate
import wandb

def linearly_decreasing_gamma(timestep, start_gamma, end_gamma, num_timesteps):
    gamma = start_gamma - (start_gamma - end_gamma) * timestep / num_timesteps
    return gamma

def linearly_increasing_gamma(timestep, start_gamma, end_gamma, num_timesteps):
    gamma = start_gamma + (end_gamma - start_gamma) * timestep / num_timesteps
    return gamma

def randomly_varying_gamma(start_gamma, end_gamma):
    gamma = np.random.uniform(start_gamma, end_gamma)
    return gamma

def main(config):
    # Fix random seed
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # CUDA setting
    torch.set_num_threads(1)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    # wandb
    if args.use_offline_wandb:
        os.environ['WANDB_MODE'] = 'dryrun'
    args.exp_id = f"{args.exp_name}_{args.env_name}_{args.use_which_gae}_{args.flag}"
    print("id", args.exp_id)
    wandb.init(project='ICDE', name=args.exp_id, entity="mingatum")
    wandb.config.update(args)


    # Create directories
    os.makedirs(config["log_dir"], exist_ok=True)
    if not config["debug"]:
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["save_dir"], exist_ok=True)

    # Create logger
    log_file = "-{}-{}-s{}".format(
        config["env_name"], config["exp_name"], config["seed"]
    )
    if config["debug"]:
        log_file += "-debug"
    logger.configure(
        dir=config["log_dir"], format_strs=["csv", "stdout"], log_suffix=log_file
    )
    print("\nLog File:", log_file)

    # Create environments
    envs = make_envs(
        num_envs=config["num_processes"],
        env_name=config["env_name"],
        num_levels=config["num_levels"],
        start_level=config["start_level"],
        distribution_mode=config["distribution_mode"],
        normalize_reward=config["normalize_reward"],
        device=device,
    )
    obs_space = envs.observation_space
    action_space = envs.action_space

    # Create actor-critic
    actor_critic_class = getattr(sys.modules[__name__], config["actor_critic_class"])
    actor_critic_params = config["actor_critic_params"]
    actor_critic = actor_critic_class(
        obs_space.shape, action_space.n, **actor_critic_params
    )
    actor_critic.to(device)
    print("\nActor-Critic Network:", actor_critic)

    # Create rollout storage
    rollouts = RolloutStorage(
        config["num_steps"], config["num_processes"], obs_space.shape, action_space
    )
    rollouts.to(device)

    # Create agent
    agent_class = getattr(sys.modules[__name__], config["agent_class"])
    agent_params = config["agent_params"]
    agent = agent_class(actor_critic, **agent_params, device=device)

    # Initialize environments
    obs = envs.reset()
    *_, infos = envs.step_wait()
    levels = torch.LongTensor([info["level_seed"] for info in infos])
    rollouts.obs[0].copy_(obs)
    rollouts.levels[0].copy_(levels)

    # Train actor-critic
    num_env_steps_epoch = config["num_steps"] * config["num_processes"]
    num_updates = int(config["num_env_steps"]) // num_env_steps_epoch
    elapsed_time = 0

    for j in range(num_updates):
        # Start training
        start = time.time()

        # Set actor-critic to train mode
        actor_critic.train()

        # Sample episode
        sample_episodes(envs, rollouts, actor_critic)

        # Compute return
        with torch.no_grad():
            next_critic_outputs = actor_critic.forward_critic(rollouts.obs[-1])
            next_value = next_critic_outputs["value"]

        if args.use_which_gae == 'normal':
            
            if j % config["log_interval"] == 0:
                # Train statistics
                now_steps = (j + 1) * config["num_processes"] * config["num_steps"]

            if config["gamma_type"] == 'decrease':
                gamma = linearly_decreasing_gamma(now_steps, config["start_gamma"], config["end_gamma"], config["num_env_steps"])
            elif config["gamma_type"] == 'increase':
                gamma = linearly_increasing_gamma(now_steps, config["start_gamma"], config["end_gamma"], config["num_env_steps"])
            elif config["gamma_type"] == 'random':
                gamma = randomly_varying_gamma(config["start_gamma"], config["end_gamma"])

            print("gamma" , gamma)
            # print("steps", now_steps)
            # print("num_env_steps", config["num_env_steps"])
            rollouts.compute_returns(next_value, gamma, config["gae_lambda"])

        elif args.use_which_gae == 'average':
            gammas = [0.80, 0.90, 0.99, 0.95, 0.999]
            rollouts.compute_average_gae_returns(next_value, gammas, config["gae_lambda"])

        elif args.use_which_gae == 'fixed':
            rollouts.compute_returns(next_value, config["gamma"], config["gae_lambda"])

        rollouts.compute_advantages()

        # Update actor-critic
        train_statistics = agent.update(rollouts)

        # Reset rollout storage
        rollouts.after_update()

        # End training
        end = time.time()
        elapsed_time += end - start

        # Statistics
        if j % config["log_interval"] == 0:
            # Train statistics
            total_num_steps = (j + 1) * config["num_processes"] * config["num_steps"]
            time_per_epoch = elapsed_time / (j + 1)

            print(
                "\nUpdate {}, step {}, time per epoch {:.2f} \n".format(
                    j, total_num_steps, time_per_epoch
                )
            )

            logger.logkv("train/total_num_steps", total_num_steps)
            logger.logkv("train/time_per_epoch", time_per_epoch)

            wandb.log({'train/total_num_steps': total_num_steps,
                       'train/time_per_epoch': time_per_epoch,
                     })
            for key, val in train_statistics.items():
                logger.logkv("train/{}".format(key), val)
                wandb.log({'train/total_num_steps': total_num_steps,
                           'train/{}'.format(key): val})

            # Fetch reward normalizing variables
            norm_infos = envs.normalization_infos()

            # Evaluate actor-critic on train environments
            train_eval_statistics, train_value_statistics = evaluate(
                config, actor_critic, device, test_envs=False, norm_infos=norm_infos
            )
            train_episode_rewards = train_eval_statistics["episode_rewards"]
            train_episode_steps = train_eval_statistics["episode_steps"]
           
            print(
                "Last {} training episodes: \n"
                "mean/med/std reward {:.2f}/{:.2f}/{:.2f}, "
                "mean/med/std step {:.2f}/{:.2f}/{:.2f} \n".format(
                    len(train_episode_rewards),
                    np.mean(train_episode_rewards),
                    np.median(train_episode_rewards),
                    np.std(train_episode_rewards),
                    np.mean(train_episode_steps),
                    np.median(train_episode_steps),
                    np.std(train_episode_steps),
                )
            )

            logger.logkv("train/mean_episode_reward", np.mean(train_episode_rewards))
            logger.logkv("train/med_episode_reward", np.median(train_episode_rewards))
            logger.logkv("train/std_episode_reward", np.std(train_episode_rewards))
            logger.logkv("train/mean_episode_step", np.mean(train_episode_steps))
            logger.logkv("train/med_episode_step", np.median(train_episode_steps))
            logger.logkv("train/std_episode_step", np.std(train_episode_steps))

            wandb.log({'total_num_steps': total_num_steps,
                       'train/mean_episode_reward': np.mean(train_episode_rewards),
                       'train/med_episode_reward': np.median(train_episode_rewards),
                       'train/std_episode_reward': np.std(train_episode_rewards),
                       'train/mean_episode_step': np.mean(train_episode_steps),
                       'train/med_episode_step': np.median(train_episode_steps),
                       'train/std_episode_step': np.std(train_episode_steps),
                   })

            for key, val in train_value_statistics.items():
                logger.logkv("train/{}".format(key), val) 
                wandb.log({'train/total_num_steps': total_num_steps,
                    'train/{}'.format(key): val})

            # Evaluate actor-critic on test environments
            test_eval_statistics, *_ = evaluate(
                config, actor_critic, device, test_envs=True
            )
            test_episode_rewards = test_eval_statistics["episode_rewards"]
            test_episode_steps = test_eval_statistics["episode_steps"]

            print(
                "Last {} test episodes: \n"
                "mean/med/std reward {:.2f}/{:.2f}/{:.2f}, "
                "mean/med/std step {:.2f}/{:.2f}/{:.2f} \n".format(
                    len(test_episode_rewards),
                    np.mean(test_episode_rewards),
                    np.median(test_episode_rewards),
                    np.std(test_episode_rewards),
                    np.mean(test_episode_steps),
                    np.median(test_episode_steps),
                    np.std(test_episode_steps),
                )
            )

            logger.logkv("test/mean_episode_reward", np.mean(test_episode_rewards))
            logger.logkv("test/med_episode_reward", np.median(test_episode_rewards))
            logger.logkv("test/std_episode_reward", np.std(test_episode_rewards))
            logger.logkv("test/mean_episode_step", np.mean(test_episode_steps))
            logger.logkv("test/med_episode_step", np.median(test_episode_steps))
            logger.logkv("test/std_episode_step", np.std(test_episode_steps))

            wandb.log({ 'total_num_steps': total_num_steps,
                        'test/mean_episode_reward': np.mean(test_episode_rewards),
                        'test/med_episode_reward': np.median(test_episode_rewards),
                        'test/std_episode_reward': np.std(test_episode_rewards),
                        'test/mean_episode_step': np.mean(test_episode_steps),
                        'test/med_episode_step': np.median(test_episode_steps),
                        'test/std_episode_step': np.std(test_episode_steps),
                     })

            logger.dumpkvs()

        if j == num_updates - 1 and not config["debug"]:
            print("\nFinal evaluation \n")

            # Evaluate actor-critic on train environments
            train_eval_statistics, *_ = evaluate(
                config, actor_critic, device, test_envs=False
            )
            train_episode_rewards = train_eval_statistics["episode_rewards"]
            train_episode_steps = train_eval_statistics["episode_steps"]

            print(
                "Last {} train episodes: \n"
                "mean/med/std reward {:.2f}/{:.2f}/{:.2f}, "
                "mean/med/std step {:.2f}/{:.2f}/{:.2f} \n".format(
                    len(train_episode_rewards),
                    np.mean(train_episode_rewards),
                    np.median(train_episode_rewards),
                    np.std(train_episode_rewards),
                    np.mean(train_episode_steps),
                    np.median(train_episode_steps),
                    np.std(train_episode_steps),
                )
            )

            # Save train scores
            np.save(
                os.path.join(config["output_dir"], "scores-train{}.npy".format(log_file)),
                np.array(train_episode_rewards),
            )

            # Evaluate actor-critic on test environments
            test_eval_statistics, *_ = evaluate(
                config, actor_critic, device, test_envs=True
            )
            test_episode_rewards = test_eval_statistics["episode_rewards"]
            test_episode_steps = test_eval_statistics["episode_steps"]

            print(
                "Last {} test episodes: \n"
                "mean/med/std reward {:.2f}/{:.2f}/{:.2f}, "
                "mean/med/std step {:.2f}/{:.2f}/{:.2f} \n".format(
                    len(test_episode_rewards),
                    np.mean(test_episode_rewards),
                    np.median(test_episode_rewards),
                    np.std(test_episode_rewards),
                    np.mean(test_episode_steps),
                    np.median(test_episode_steps),
                    np.std(test_episode_steps),
                )
            )

            # Save test scores
            np.save(
                os.path.join(config["output_dir"], "scores-test{}.npy".format(log_file)),
                np.array(test_episode_rewards),
            )

            # Save checkpoint
            torch.save(
                actor_critic.state_dict(),
                os.path.join(config["save_dir"], "agent{}.pt".format(log_file)),
            )


if __name__ == "__main__":
    # Argument
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--flag", type=str, default='f')

    parser.add_argument('--use_offline_wandb', action='store_true', help='use offline wandb')

    parser.add_argument('--use_which_gae', type=str, default='normal', choices=['average', 'normal', 'fixed'], help='Just use normal gae can use gamma_type and start_gamma, end_gamma ')
    parser.add_argument('--gamma_type', type=str, default='random', choices=['increase', 'decrease', 'random'])
    parser.add_argument('--start_gamma', type=float, default=0.95)
    parser.add_argument('--end_gamma', type=float, default=0.99)

    args = parser.parse_args()

    # Load config
    config_file = open("configs/{}.yaml".format(args.exp_name), "r")
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Update config
    config["exp_name"] = args.exp_name
    config["env_name"] = args.env_name
    config["seed"] = args.seed
    config["debug"] = args.debug
    config["gamma_type"] = args.gamma_type
    config["start_gamma"] = args.start_gamma
    config["end_gamma"] = args.end_gamma

    # Run main
    main(config)
