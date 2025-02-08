# noqa: D212, D415
"""
# Simple Adversary

```{figure} mpe_simple_adversary.gif
:width: 140px
:name: simple_adversary
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_adversary_v3` |
|--------------------|--------------------------------------------------|
| Actions            | Discrete/Continuous                              |
| Parallel API       | Yes                                              |
| Manual Control     | No                                               |
| Agents             | `agents= [adversary_0, agent_0,agent_1]`         |
| Agents             | 3                                                |
| Action Shape       | (5)                                              |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5))                   |
| Observation Shape  | (8),(10)                                         |
| Observation Values | (-inf,inf)                                       |
| State Shape        | (28,)                                            |
| State Values       | (-inf,inf)                                       |


In this environment, there is 1 adversary (red), N good agents (green), N landmarks (default N=2). All agents observe the position of landmarks and other agents. One landmark is the 'target landmark' (colored green). Good agents are rewarded based on how close the closest one of them is to the
target landmark, but negatively rewarded based on how close the adversary is to the target landmark. The adversary is rewarded based on distance to the target, but it doesn't know which landmark is the target landmark. All rewards are unscaled Euclidean distance (see main MPE documentation for
average distance). This means good agents have to learn to 'split up' and cover all landmarks to deceive the adversary.

Agent observation space: `[goal_rel_position, landmark_rel_position, other_agent_rel_positions]`

Adversary observation space: `[landmark_rel_position, other_agents_rel_positions]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

Adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=False, dynamic_rescaling=False)
```



`N`:  number of good agents and landmarks

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen size

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "simple_adversary_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

TIME_PENALTY = 0.01           # A small per-step cost (good team gets -TIME_PENALTY, adversaries get +TIME_PENALTY)
COLLISION_THRESHOLD = 0.5     # Distance below which a collision (or interference) is counted
COLLISION_MAG = 1.0           # Magnitude for each collision event
DIRECTIONAL_MULTIPLIER = 0.1  # Scales the directional alignment bonus

def directional_bonus(agent, goal_pos):
    """
    Compute the cosine similarity between the agent's velocity and the vector from its position to the goal.
    This yields a bonus in the range [-1, 1] where 1 means perfect alignment.
    """
    vel = agent.state.p_vel
    pos = agent.state.p_pos
    direction = goal_pos - pos
    norm_direction = np.linalg.norm(direction)
    norm_vel = np.linalg.norm(vel)
    if norm_direction == 0 or norm_vel == 0:
        return 0.0
    return np.dot(vel, direction) / (norm_direction * norm_vel) 

class Scenario(BaseScenario):
    def make_world(self, N=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_adversaries = 2  # Update number of adversaries to 2
        num_good_agents = 2  # Update number of good agents to 2
        num_landmarks = 3    # Update number of landmarks to 3
        num_agents = num_adversaries + num_good_agents
        world.num_agents = num_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark_{i}"
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        return world

    def reset_world(self, world, np_random):
        # Random properties for agents
        for i, agent in enumerate(world.agents):
            if agent.adversary:
                agent.color = np.array([0.85, 0.35, 0.35])  # Red for adversaries
            else:
                agent.color = np.array([0.35, 0.85, 0.35])  # Green for good agents

        # Random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])  # Default gray for landmarks

        # Set goal landmark and update color
        goal = np_random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])  # Green for target landmark
        for agent in world.agents:
            agent.goal_a = goal

        # Set random initial states for agents
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # Set random initial states for landmarks
        for landmark in world.landmarks:
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for lm in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - lm.state.p_pos)))
            dists.append(
                np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
            )
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        """
        Dispatch reward based on agent type.
        """
        if agent.adversary:
            return self.adversary_reward(agent, world)
        else:
            return self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        """
        Compute the reward for good agents.
        
        Components:
          1. Distance-based term: good team reward is
                R_distance = - min_{good}(d(good, goal)) + sum_{adv}(d(adv, goal))
             so that if any good agent gets closer to the goal (smaller min distance)
             the reward increases, while adversaries being far increases the reward.
          
          2. Collision penalty: For every pair (good, adv) closer than COLLISION_THRESHOLD,
             subtract COLLISION_MAG.
          
          3. Directional alignment bonus: Reward good agents for having velocity aligned
             toward their goal.
          
          4. Time penalty: A small constant penalty each timestep.
          
        All terms have corresponding opposites in the adversary reward.
        """
        # --- (1) Distance-based term ---
        good_agents = self.good_agents(world)
        # Use Euclidean norm for distance
        min_good_distance = min(
            np.linalg.norm(a.state.p_pos - a.goal_a.state.p_pos)
            for a in good_agents
        )
        adversary_agents = self.adversaries(world)
        total_adv_distance = sum(
            np.linalg.norm(a.state.p_pos - a.goal_a.state.p_pos)
            for a in adversary_agents
        )
        distance_reward = -min_good_distance + total_adv_distance

        # --- (2) Collision / interference term ---
        collision_penalty = 0.0
        for g in good_agents:
            for adv in adversary_agents:
                d = np.linalg.norm(g.state.p_pos - adv.state.p_pos)
                if d < COLLISION_THRESHOLD:
                    collision_penalty -= COLLISION_MAG

        # --- (3) Directional alignment bonus ---
        directional_reward = 0.0
        # We assume each good agent’s goal is defined in its 'goal_a'
        for a in good_agents:
            goal_pos = a.goal_a.state.p_pos
            directional_reward += directional_bonus(a, goal_pos)
        directional_reward *= DIRECTIONAL_MULTIPLIER

        # --- (4) Time penalty ---
        # Here we subtract a small constant so that good agents are encouraged to act quickly.
        time_term = -TIME_PENALTY

        total_reward = distance_reward + collision_penalty + directional_reward + time_term
        return total_reward

    def adversary_reward(self, agent, world):
        """
        Compute the reward for adversaries.
        
        Components:
          1. Distance-based term (the negative of the good team term):
                R_distance_adv = min_{good}(d(good, goal)) - sum_{adv}(d(adv, goal))
          
          2. Collision bonus: For every (good, adv) pair closer than threshold,
             add COLLISION_MAG.
          
          3. Directional alignment penalty: Adversaries are rewarded if good agents are 
             not moving well toward the goal (i.e. subtract the directional bonus).
          
          4. Time bonus: Here we add TIME_PENALTY, the exact opposite of the good team.
          
        These components cancel with those in agent_reward to maintain zero-sum.
        """
        # --- (1) Distance-based term ---
        good_agents = self.good_agents(world)
        min_good_distance = min(
            np.linalg.norm(a.state.p_pos - a.goal_a.state.p_pos)
            for a in good_agents
        )
        adversary_agents = self.adversaries(world)
        total_adv_distance = sum(
            np.linalg.norm(a.state.p_pos - a.goal_a.state.p_pos)
            for a in adversary_agents
        )
        distance_reward_adv = min_good_distance - total_adv_distance

        # --- (2) Collision bonus ---
        collision_bonus = 0.0
        for g in good_agents:
            for adv in adversary_agents:
                d = np.linalg.norm(g.state.p_pos - adv.state.p_pos)
                if d < COLLISION_THRESHOLD:
                    collision_bonus += COLLISION_MAG

        # --- (3) Directional alignment penalty ---
        directional_penalty = 0.0
        for a in good_agents:
            goal_pos = a.goal_a.state.p_pos
            directional_penalty += directional_bonus(a, goal_pos)
        directional_penalty *= DIRECTIONAL_MULTIPLIER

        # --- (4) Time bonus ---
        # Note: This is the opposite of the good team's time penalty.
        time_term = +TIME_PENALTY

        total_reward = distance_reward_adv + collision_bonus - directional_penalty + time_term
        return total_reward

    # def agent_reward(self, agent, world):
    #     # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
    #     shaped_reward = True
    #     shaped_adv_reward = True

    #     # Calculate negative reward for adversary
    #     adversary_agents = self.adversaries(world)
    #     if shaped_adv_reward:  # distance-based adversary reward
    #         adv_rew = sum(
    #             np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
    #             for a in adversary_agents
    #         )
    #     else:  # proximity-based adversary reward (binary)
    #         adv_rew = 0
    #         for a in adversary_agents:
    #             if (
    #                 np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
    #                 < 2 * a.goal_a.size
    #             ):
    #                 adv_rew -= 5

    #     # Calculate positive reward for agents
    #     good_agents = self.good_agents(world)
    #     if shaped_reward:  # distance-based agent reward
    #         pos_rew = -min(
    #             np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
    #             for a in good_agents
    #         )
    #     else:  # proximity-based agent reward (binary)
    #         pos_rew = 0
    #         if (
    #             min(
    #                 np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
    #                 for a in good_agents
    #             )
    #             < 2 * agent.goal_a.size
    #         ):
    #             pos_rew += 5
    #         pos_rew -= min(
    #             np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
    #             for a in good_agents
    #         )
    #     return pos_rew + adv_rew

    # def adversary_reward(self, agent, world):
    #     # Rewarded based on proximity to the goal landmark
    #     shaped_reward = True
    #     if shaped_reward:  # distance-based reward
    #         return -np.sqrt(
    #             np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
    #         )
    #     else:  # proximity-based reward (binary)
    #         adv_rew = 0
    #         if (
    #             np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
    #             < 2 * agent.goal_a.size
    #         ):
    #             adv_rew += 5
    #         return adv_rew

    # def agent_reward(self, agent, world):
    #     # Normalize distances
    #     max_distance = np.sqrt(2) * 2  # Assuming world bounds are [-1, 1] in each dimension
    #     adversary_agents = self.adversaries(world)
    #     good_agents = self.good_agents(world)

    #     # Positive reward for good agents (normalized distance to goal)
    #     pos_rew = -min(
    #         np.linalg.norm(a.state.p_pos - a.goal_a.state.p_pos) / (max_distance + 1)
    #         for a in good_agents
    #     )

    #     # Negative reward for adversaries (normalized distance to goal)
    #     adv_rew = max(
    #         np.linalg.norm(a.state.p_pos - a.goal_a.state.p_pos) / (max_distance + 1)
    #         for a in adversary_agents
    #     )

    #     return pos_rew + adv_rew

    # def adversary_reward(self, agent, world):
    #     # Adversary reward is the negative of agent reward
    #     return -self.agent_reward(agent, world)


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            return np.concatenate(
                [agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos
            )
        else:
            return np.concatenate(entity_pos + other_pos)
