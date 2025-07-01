from .env import Env
from .mo_env import MOEnv
from .recorder import Recorder
from .wrappers import LinearReward

try:
  import gymnasium as gym
  gym.register(
      id='CrafterReward-v1',
      entry_point='crafter:Env',
      max_episode_steps=10000,
      kwargs={'reward': True})
  gym.register(
      id='CrafterNoReward-v1',
      entry_point='crafter:Env',
      max_episode_steps=10000,
      kwargs={'reward': False})
  gym.register(
      id='CrafterMOReward-v1',
      entry_point='crafter:MOEnv',
      max_episode_steps=10000,
      kwargs={'reward': True})
except ImportError:
  pass

