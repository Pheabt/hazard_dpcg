from typing import Iterator, Sequence, Tuple

import torch
from torch import device, Tensor
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from gym.spaces import Space


class RolloutStorage(object):
    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        obs_shape: Sequence[int],
        action_space: Space,
    ):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.levels = torch.LongTensor(num_steps + 1, num_processes).fill_(0)

        self.num_steps = num_steps
        self.step = 0

    def __getitem__(self, key: str):
        return getattr(self, key)

    def to(self, device: device):
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.masks = self.masks.to(device)
        self.levels = self.levels.to(device)

    def insert(
        self,
        obs: Tensor,
        actions: Tensor,
        action_log_probs: Tensor,
        rewards,
        value_preds: Tensor,
        masks: Tensor,
        levels: Tensor,
    ):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.rewards[self.step].copy_(rewards)
        self.value_preds[self.step].copy_(value_preds)
        self.masks[self.step + 1].copy_(masks)
        self.levels[self.step + 1].copy_(levels)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.levels[0].copy_(self.levels[-1])

    def compute_returns(self, next_value: Tensor, gamma: float, gae_lambda: float):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = (
                self.rewards[step]
                + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                - self.value_preds[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]
            #print('.................normal_self.returns', self.returns)



    # def compute_returns(self, next_value: Tensor, gamma: float, gae_lambda: float):
    #     self.value_preds[-1] = next_value
    #     for step in reversed(range(self.rewards.size(0))):
    #         delta = (
    #                 self.rewards[step]
    #                 + gamma * self.value_preds[step + 1] * self.masks[step + 1]
    #                 - self.value_preds[step]
    #         )
    #         self.returns[step] = delta + self.value_preds[step]
    #         # print('.................normal_self.returns', self.returns)





    def compute_average_gae_returns(self, next_value: Tensor, gammas: list, gae_lambda: float):
        self.value_preds[-1] = next_value
        num_steps = self.rewards.size(0)
        num_gammas = len(gammas)
        returns_avg = torch.zeros_like(self.returns)

        # 对每个gamma值计算GAE
        for gamma in gammas:
            gae = 0
            for step in reversed(range(num_steps)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
            # 累加所有gamma值的returns
            returns_avg[:num_steps] += self.returns[:num_steps]

        # 计算平均returns
        self.returns = returns_avg / num_gammas
        #print('.................average_self.returns', self.returns)




    def compute_advantages(self):
        self.advantages = self.returns[:-1] - self.value_preds[:-1]
        mean = self.advantages.mean()
        std = self.advantages.std()
        self.advantages = (self.advantages - mean) / (std + 1e-5)







    def feed_forward_generator(
        self,
        num_mini_batch: int = None,
        mini_batch_size: int = None,
    ) -> Iterator[Tuple[Tensor, ...]]:
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, "[ERROR] num mini batch is too large"
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True,
        )

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            next_obs_batch = self.obs[1:].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            rewards_batch = self.rewards.view(-1, 1)[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            returns_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[1:].view(-1, 1)[indices]
            levels_batch = self.levels[:-1].view(-1, 1)[indices]

            if hasattr(self, "advantages"):
                adv_targs = self.advantages.view(-1, 1)[indices]
            else:
                adv_targs = None

            yield (
                obs_batch,
                next_obs_batch,
                actions_batch,
                old_action_log_probs_batch,
                rewards_batch,
                value_preds_batch,
                returns_batch,
                masks_batch,
                adv_targs,
                levels_batch,
            )
