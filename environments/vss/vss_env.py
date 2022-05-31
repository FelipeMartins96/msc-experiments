import functools
import random

import numpy as np
from gym.spaces import Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from rsoccer_gym.Simulators.rsim import RSimVSS
from rsoccer_gym.Entities import Frame, Ball, Robot
from rsoccer_gym.Utils import KDTree

def actions_to_v_wheels(actions):
    left_wheel_speed = actions[0]
    right_wheel_speed = actions[1]

    left_wheel_speed, right_wheel_speed = np.clip(
        (left_wheel_speed, right_wheel_speed), -1, 1
    )

    # Convert to rad/s
    left_wheel_speed /= 0.026
    right_wheel_speed /= 0.026

    return left_wheel_speed , right_wheel_speed

def get_initial_frame(sim):
    """Returns the position of each robot and ball for the initial frame"""
    field = sim.get_field_params()
    field_half_length = field.length / 2
    field_half_width = field.width / 2

    def x():
        return random.uniform(-field_half_length + 0.1, field_half_length - 0.1)

    def y():
        return random.uniform(-field_half_width + 0.1, field_half_width - 0.1)

    def theta():
        return random.uniform(0, 360)

    pos_frame: Frame = Frame()

    pos_frame.ball = Ball(x=x(), y=y())

    min_dist = 0.1

    places = KDTree()
    places.insert((pos_frame.ball.x, pos_frame.ball.y))

    for i in range(sim.n_robots_blue):
        pos = (x(), y())
        while places.get_nearest(pos)[1] < min_dist:
            pos = (x(), y())

        places.insert(pos)
        pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

    for i in range(sim.n_robots_yellow):
        pos = (x(), y())
        while places.get_nearest(pos)[1] < min_dist:
            pos = (x(), y())

        places.insert(pos)
        pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

    return pos_frame


def update_observations(sim, agents):
    frame = sim.get_frame()

    def obs_from_frame(agent):
        team, idx = agent.split("_")
        idx = int(idx)
        allies = frame.robots_blue
        enemies = frame.robots_yellow

        if team == "yellow":
            allies, enemies = enemies, allies

        observation = []

        observation.append(frame.ball.x)
        observation.append(frame.ball.y)
        observation.append(frame.ball.v_x)
        observation.append(frame.ball.v_y)

        allies_keys = list(allies.keys())
        allies_keys[0], allies_keys[idx] = allies_keys[idx], allies_keys[0]
        for k in allies_keys:
            ally = allies[k]
            observation.append(ally.x)
            observation.append(ally.y)
            observation.append(np.sin(np.deg2rad(ally.theta)))
            observation.append(np.cos(np.deg2rad(ally.theta)))
            observation.append(ally.v_x)
            observation.append(ally.v_y)
            observation.append(ally.v_theta)

        for enemy in enemies.values():
            observation.append(enemy.x)
            observation.append(enemy.y)
            observation.append(enemy.v_x)
            observation.append(enemy.v_y)
            observation.append(enemy.v_theta)

    return {agent: obs_from_frame(agent) for agent in agents}


def env(**kwargs):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env(**kwargs)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gym, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "vss_v0"}

    def __init__(self, n_robots_blue=3, n_robots_yellow=3):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        self.sim = RSimVSS(
            field_type=0,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            time_step_ms=25,
        )
        self.view = None

        self.possible_agents = ["blue_" + str(r) for r in range(n_robots_blue)] + [
            "yellow_" + str(r) for r in range(n_robots_yellow)
        ]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self._action_spaces = {
            agent: Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: Box(low=-1, high=1, shape=(40,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return self._action_spaces[agent]

    def action_space(self, agent):
        return self._observation_spaces[agent]

    def render(self, mode="human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.view == None:
            from rsoccer_gym.Render import RCGymRender

            self.view = RCGymRender(
                self.sim.n_robots_blue,
                self.sim.n_robots_yellow,
                self.sim.get_field_params(),
                simulator="vss",
            )

        return self.view.render_frame(
            self.sim.get_frame(), return_rgb_array=mode == "rgb_array"
        )

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.commands = []

        # Reset robot positions
        initial_pos_frame = get_initial_frame(self.sim)
        self.sim.reset(initial_pos_frame)

        self.observations = update_observations(self.sim, self.agents)
        self.num_moves = 0

        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        agent = self.agent_selection
        
        v_wheel0, v_wheel1 = actions_to_v_wheels(action)
        self.commands.append(Robot(yellow='yellow' in agent, id=int(agent[-1]), v_wheel0=v_wheel0, v_wheel1=v_wheel1))
        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            self.sim.send_commands(self.commands)
            self.commands = []
            # rewards for all agents are placed in the .rewards dictionary
            self.rewards = {agent: -1 for agent in self.agents}

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.num_moves >= 100 for agent in self.agents}

            # observe the current state
            self.observations = update_observations(self.sim, self.agents)
        else:
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
