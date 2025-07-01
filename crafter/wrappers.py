import gymnasium as gym
import numpy as np
from typing import Tuple, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class LinearReward(gym.Wrapper):
    """Makes the env return a scalar reward, which is the dot-product between the reward vector and the weight vector."""

    def __init__(self, env: gym.Env, weight: np.ndarray = None):
        """Makes the env return a scalar reward, which is the dot-product between the reward vector and the weight vector.

        Args:
            env: env to wrap
            weight: weight vector to use in the dot product
        """
        gym.utils.RecordConstructorArgs.__init__(self, weight=weight)
        gym.Wrapper.__init__(self, env)
        if weight is None:
            weight = np.ones(shape=env.unwrapped.reward_space.shape)
        self.set_weight(weight)

    def set_weight(self, weight: np.ndarray):
        """Changes weights for the scalarization.

        Args:
            weight: new weights to set
        Returns: nothing
        """
        assert weight.shape == self.env.unwrapped.reward_space.shape, "Reward weight has different shape than reward vector."
        self.w = weight

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Steps in the environment.

        Args:
            action: action to perform
        Returns: obs, scalarized_reward, terminated, truncated, info
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        scalar_reward = np.dot(reward, self.w)
        info["vector_reward"] = reward
        info["reward_weights"] = self.w

        return observation, scalar_reward, terminated, truncated, info