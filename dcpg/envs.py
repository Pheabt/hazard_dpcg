from typing import Dict, Union

from numpy import ndarray
import torch
from torch import device, Tensor

from baselines.common.vec_env import (
    VecEnv,
    VecEnvWrapper,
    VecExtractDictObs,
    VecMonitor,
    VecNormalize,
)
from gym.spaces.box import Box
from procgen import ProcgenEnv


class VecPyTorchProcgen(VecEnvWrapper):
    def __init__(self, venv: VecEnv, device: device):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        super(VecPyTorchProcgen, self).__init__(venv)
        self.device = device

        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [3, 64, 64],
            dtype=self.observation_space.dtype,
        )

    def reset(self):
        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.0
        return obs

    def step_async(self, actions: Tensor):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.0
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def normalization_infos(self) -> Dict[str, Union[float, ndarray]]:
        var = self.venv.ret_rms.var
        eps = self.venv.epsilon
        cliprew = self.venv.cliprew
        norm_infos = {
            "var": var,
            "eps": eps,
            "cliprew": cliprew,
        }
        return norm_infos


def make_envs(
    num_envs: int,
    env_name: str,
    num_levels: int,
    start_level: int,
    distribution_mode: str,
    normalize_reward: bool,
    device: device,
) -> VecPyTorchProcgen:
    envs = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
    )
    envs = VecExtractDictObs(envs, "rgb")
    envs = VecMonitor(envs, filename=None, keep_buf=100)
    envs = VecNormalize(envs, ob=False, ret=normalize_reward)
    envs = VecPyTorchProcgen(envs, device)

    return envs
