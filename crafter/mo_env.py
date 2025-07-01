import numpy as np
from typing import Any, Union

from . import constants
from . import objects
from . import worldgen
from . import env

def compute_reward(prev_achivements: dict[Any, int], current_achievements: dict[Any, int], player: objects.Player, last_health: int, ep_player_pos: list[tuple[int, int]], prev_unlocked: set[str]) -> np.ndarray:
    """
    Compute the multi-objective reward based on new achievements and player position.

    # Reward structure:
    Dimension 0: New achievements (+1 for each new achievement) - Same as crafter reward
    Dimension 1: Health update (normalized) - Same as crafter reward
    Dimension 2: Novel player positions (+0.1 for each unique position) - Encourages exploration
    Dimension 3: Skeleton or Zombie killed (+1 for each skeleton killed) - Encourages combat
    Dimension 4: Cow killed (+1 for each cow killed) - Encourages resource gathering
    Dimension 5: Plant Eaten (+1 for each plant eaten) - Encourages resource gathering
    """

    reward = np.zeros(6)

    # Dimension 0: New achievements
    unlocked = {
        name for name, count in player.achievements.items()
        if count > 0 and name not in prev_unlocked}
    if unlocked:
      reward[0] = 1.0

    # Dimension 1: Health update
    health_change = player.health - last_health
    reward[1] = health_change / 10.0  # Normalize health change

    # Dimension 2: Novel player positions
    unique_position = not any(np.array_equal(player.pos, pos) for pos in ep_player_pos)
    reward[2] = 0.1 if unique_position else 0.0

    # Dimension 3: Skeleton or Zombie killed
    zombie_killed = current_achievements.get('defeat_zombie', 0) - prev_achivements.get('defeat_zombie', 0)
    skeleton_killed = current_achievements.get('defeat_skeleton', 0) - prev_achivements.get('defeat_skeleton', 0)
    reward[3] = 1.0 if zombie_killed > 0 or skeleton_killed > 0 else 0.0

    # Dimension 4: Cow killed
    cow_killed = current_achievements.get('eat_cow', 0) - prev_achivements.get('eat_cow', 0)
    reward[4] = 1.0 if cow_killed > 0 else 0.0

    # Dimension 5: Plant Eaten (More complex)
    plant_eaten = current_achievements.get('eat_plant', 0) - prev_achivements.get('eat_plant', 0)
    reward[5] = 1.0 if plant_eaten > 0 else 0.0
   
    reward = np.array(reward)

    return reward


class MOEnv(env.Env):

  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64),
      reward=True, length=10000, seed=None, linearization_weights=None):
    
    """
    Multi-objective version of the Crafter environment.
    """
    super().__init__(area=area, view=view, size=size, reward=reward, length=length, seed=seed)
    self._linearization_weights = linearization_weights
    self._ep_player_pos = None

  def reset(self):
    center = (self._world.area[0] // 2, self._world.area[1] // 2)
    self._episode += 1
    self._step = 0
    self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
    self._update_time()
    self._player = objects.Player(self._world, center)
    self._last_health = self._player.health
    self._world.add(self._player)
    self._unlocked = set()
    worldgen.generate_world(self._world, self._player)

    # Update list of player positions for multi-objective optimization
    self._ep_player_pos = []
    self._ep_player_pos.append(self._player.pos)

    return self._obs()

  def step(self, action):
    self._step += 1
    self._update_time()

    # Get prev achievements dict
    prev_achivements = self._player.achievements.copy()

    # Act
    self._player.action = constants.actions[action]
    for obj in self._world.objects:
      if self._player.distance(obj) < 2 * max(self._view):
        obj.update()
    if self._step % 10 == 0:
      for chunk, objs in self._world.chunks.items():
        # xmin, xmax, ymin, ymax = chunk
        # center = (xmax - xmin) // 2, (ymax - ymin) // 2
        # if self._player.distance(center) < 4 * max(self._view):
        self._balance_chunk(chunk, objs)
    obs = self._obs()

    # Compute MO reward
    mo_reward = compute_reward(
        prev_achivements=prev_achivements,
        current_achievements=self._player.achievements,
        player=self._player,
        last_health=self._last_health,
        ep_player_pos=self._ep_player_pos,
        prev_unlocked=self._unlocked
    )

    # If linearization weights are provided, compute the linearized reward
    if self._linearization_weights is not None:
      reward = np.dot(mo_reward, self._linearization_weights)
    else:
      reward = mo_reward

    # Update player position, health and unlocked achievements
    self._ep_player_pos.append(self._player.pos)
    self._last_health = self._player.health
    unlocked = {
        name for name, count in self._player.achievements.items()
        if count > 0 and name not in self._unlocked}
    if unlocked:
      self._unlocked |= unlocked
    
    dead = self._player.health <= 0
    over = self._length and self._step >= self._length

    # Compatibility with OpenAI Gym API
    terminated = dead
    truncated = over and not dead

    info = {
        'inventory': self._player.inventory.copy(),
        'achievements': self._player.achievements.copy(),
        'discount': 1 - float(dead),
        'semantic': self._sem_view(),
        'player_pos': self._player.pos,
        'mo_reward': mo_reward,
        'reward': reward,
    }

    return obs, reward, terminated, truncated, info