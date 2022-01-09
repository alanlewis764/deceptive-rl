import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Tuple
from .env_generator import EnvironmentCollection, Environment
from gym_extensions.continuous.gym_navigation_2d.env_utils import Obstacle, Environment
from math import pi, cos, sin
import numpy as np
import pyvisgraph as vg

# from gym.envs.classic_control.rendering import make_circle, Transform

import os
import logging


class LimitedRangeBasedPOMDPNavigation2DEnv(gym.Env):
    logger = logging.getLogger(__name__)
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 worlds_pickle_filename=os.path.join(os.path.dirname(__file__), "assets", "worlds_640x480_v0.pkl"),
                 world_idx=0,
                 initial_position=np.array([1.0, 1.0]),
                 candidate_goals=None,
                 max_observation_range=1000.0,
                 destination_tolerance_range=0.5,
                 max_speed=1,
                 max_x=640.0,
                 min_x=0.0,
                 max_y=480.0,
                 min_y=0.0,
                 viewport_dilation=10,
                 goal_bonus=0,
                 max_episode_steps=2500,
                 randomised_start=False,
                 terminate_at_any_goal=False,
                 add_self_position_to_observation=False,
                 add_goal_position_to_observation=False,
                 goal_name='rg',
                 obstacle_list=None,
                 visibility_graph=None,
                 value_tables=None,
                 reward_type='value_table'):

        if obstacle_list is None:
            self.obstacle_list = []
        if candidate_goals is None:
            candidate_goals = {}
        if type(candidate_goals) == list:
            candidate_goals = {candidate_goal[2]: np.array([candidate_goal[0], candidate_goal[1]]) for candidate_goal
                               in candidate_goals}

        self.visibility_graph = visibility_graph
        self.value_tables = value_tables
        self.reward_type = reward_type
        self.candidate_goals = candidate_goals
        self.max_x = max_x
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y

        x_range = [int(min_x), int(max_x)]
        y_range = [int(min_y), int(max_y)]

        self.randomised_start = randomised_start
        self.terminate_at_any_goal = terminate_at_any_goal
        self.max_episode_steps = max_episode_steps
        self.current_episode_steps = 0

        self.goal_bonus = goal_bonus
        self.goal_bonus_dict = {name: self.goal_bonus for name in self.candidate_goals.keys()}
        self.goal_name = goal_name

        width_range = [100, 300]
        height_range = [100, 500]
        self.viewport_dilation = viewport_dilation
        self.world = Environment(x_range, y_range, obstacle_list)

        assert not (self.candidate_goals is None)

        if randomised_start:
            start_x = np.random.uniform(min_x, max_x)
            start_y = np.random.uniform(min_y, max_y)
            self.init_position = np.array([start_x, start_y])
        else:
            self.init_position = initial_position
        self.state = self.init_position.copy()

        self.max_observation_range = max_observation_range
        self.destination_tolerance_range = destination_tolerance_range
        self.viewer = None
        self.num_beams = 16
        self.max_speed = max_speed
        self.add_self_position_to_observation = add_self_position_to_observation
        self.add_goal_position_to_observation = add_goal_position_to_observation

        low = np.array([-self.max_speed, -2 * pi])
        high = np.array([self.max_speed, 2 * pi])
        self.action_space = Box(low, high)  # Tuple( (Box(0.0, self.max_speed, (1,)), Box(0.0, 2*pi, (1,))) )
        low = [-1.0] * self.num_beams
        high = [self.max_observation_range] * self.num_beams
        if add_self_position_to_observation:
            low.extend([-10000., -10000.])  # x and y coords
            high.extend([10000., 10000.])
        if add_goal_position_to_observation:
            low.extend([-10000., -10000.])  # x and y coords
            high.extend([10000., 10000.])

        self.observation_space = Box(np.array(low), np.array(high))
        self.observation = []

    def get_observation(self, state):
        delta_angle = 2 * pi / self.num_beams
        ranges = [self.world.raytrace(self.state,
                                      i * delta_angle,
                                      self.max_observation_range) for i in range(self.num_beams)]

        ranges = np.array(ranges)
        if self.add_self_position_to_observation:
            ranges = np.concatenate([ranges, self.state])
        if self.add_goal_position_to_observation:
            ranges = np.concatenate([ranges, self.candidate_goals['rg']])
        return ranges

    def step(self, action):
        old_state = self.state.copy()
        v = action[0]
        theta = action[1]
        dx = v * cos(theta)
        dy = v * sin(theta)

        new_x = self.clamp(value=self.state[0] + dx, min_val=self.min_x, max_val=self.max_x)
        new_y = self.clamp(value=self.state[1] + dy, min_val=self.min_y, max_val=self.max_y)

        self.state = np.array([new_x, new_y])

        done = False
        info = {}

        if self.max_episode_steps == self.current_episode_steps + 1:
            done = True
        self.current_episode_steps += 1

        if not self.world.point_is_in_free_space(self.state[0], self.state[1], epsilon=0.25):
            self.state = old_state  # if we hit an obstacle just stay in the same state

        reward_dict = self.reward()
        for name, coords in self.candidate_goals.items():
            if np.linalg.norm(coords - self.state) < self.destination_tolerance_range:
                reward_dict[name] += self.goal_bonus_dict[name]  # for reaching the goal
                # print("reward dict: ", reward_dict)
                self.goal_bonus_dict[name] = 0
                if name == self.goal_name or self.terminate_at_any_goal:
                    done = True

        self.observation = self.get_observation(self.state)
        return self.observation, reward_dict, done, info

    def reset(self, random=False):
        if self.randomised_start or random:
            self.init_position = self.get_init_pos()
        self.state = self.init_position
        self.current_episode_steps = 0
        self.goal_bonus_dict = {name: self.goal_bonus for name in self.candidate_goals.keys()}
        return self.get_observation(self.state)

    def get_init_pos(self):
        x = np.random.uniform(self.min_x, self.max_x)
        y = np.random.uniform(self.min_y, self.max_y)
        while not self.world.point_is_in_free_space(x, y, epsilon=0.25):
            x = np.random.uniform(self.min_x, self.max_x)
            y = np.random.uniform(self.min_y, self.max_y)
        return np.array([x, y])

    def reward(self):
        if self.reward_type == 'value_table':
            return self.value_table_reward()
        elif self.reward_type == 'distance':
            return self.distance_reward()
        elif self.reward_type == 'graph':
            return self.graph_reward()
        else:
            return self.time_reward()

    def time_reward(self):
        reward_dict = {}
        for name, coords in self.candidate_goals.items():
            reward_dict[name] = -1
        return reward_dict

    def graph_reward(self, done=False):
        reward_dict = {}
        if done:
            for name, coords in self.candidate_goals.items():
                current_point = vg.Point(self.state[0], self.state[1])
                goal_point = vg.Point(coords[0], coords[1])
                optpath = self.visibility_graph.shortest_path(current_point, goal_point)
                optcost = self.path_to_cost(optpath)
                reward_dict[name] = 100 - optcost
        else:
            for name, coords in self.candidate_goals.items():
                reward_dict[name] = 0
        return reward_dict

    def value_table_reward(self):
        reward_dict = {}
        for name, coords in self.candidate_goals.items():
            value_table = self.value_tables[name]
            value = value_table[int(self.state[1])][int(self.state[0])]
            reward_dict[name] = value / 100
        # print(f'{self.state}: {reward_dict["rg"]}')
        return reward_dict

    def distance_reward(self):
        reward_dict = {}
        for name, coords in self.candidate_goals.items():
            distance_from_goal = self.distance(coords, self.state)
            distance_goal_from_start = self.distance(coords, self.init_position)
            reward_dict[name] = -distance_from_goal / distance_goal_from_start
        return reward_dict

    def path_to_cost(self, path):
        prev_point = path[0]
        distance = 0
        for point in path[1:]:
            distance += self.l2norm(prev_point.x, prev_point.y, point.x, point.y)
            prev_point = point
        return distance

    @staticmethod
    def distance(s_1, s_2):
        x_1, y_1 = s_1
        x_2, y_2 = s_2
        return np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

    @staticmethod
    def l2norm(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def clamp(value, min_val, max_val):
        return min(max(value, min_val), max_val)


class StateBasedMDPNavigation2DEnv(LimitedRangeBasedPOMDPNavigation2DEnv):
    logger = logging.getLogger(__name__)

    def __init__(self, *args, **kwargs):
        LimitedRangeBasedPOMDPNavigation2DEnv.__init__(self, *args, **kwargs)
        low = [-float('inf'), -float('inf')]
        high = [float('inf'), float('inf')]

        if self.add_goal_position_to_observation:
            low.extend([-10000., -10000.])  # x and y coords
            high.extend([10000., 10000.])

        self.observation_space = Box(np.array(low), np.array(high))

    @classmethod
    def load_from_file(cls, fp, optcost, start_pos, real_goal, fake_goals, random_start=False,
                       terminate_at_any_goal=True, goal_name='rg', viewport_dilation=10,
                       destination_tolerance_range=0.5, max_speed=1, max_episode_steps=2500, goal_bonus=0,
                       visibility_graph=None, value_tables=None, reward_type='value_table'):
        with open(fp, 'r') as f:
            type = f.readline().split(" ")[1]
            height = int(f.readline().split(" ")[1])
            width = int(f.readline().split(" ")[1])
            f.readline()  # read the map heading
            map_array = []
            for y in range(height):
                map_array.append(f.readline())

        obstacle_list = []
        for y in range(height):
            for x in range(width):
                cell = map_array[y][x]
                if cell == 'T':
                    # read the obstacle one cell at a time to match up with the discrete maps
                    # the cell is read in as the top left corner
                    obstacle_list.append(Obstacle(
                        c=np.array([x + 0.5, y + 0.5]),
                        h=1.,
                        w=1.
                    ))

        candidate_goals = fake_goals + [real_goal]
        return cls(
            initial_position=np.array(start_pos),
            candidate_goals=candidate_goals,
            randomised_start=random_start,
            terminate_at_any_goal=terminate_at_any_goal,
            goal_name=goal_name,
            obstacle_list=obstacle_list,
            viewport_dilation=viewport_dilation,
            max_x=float(width),
            min_x=0.0,
            max_y=float(height),
            min_y=0.0,
            destination_tolerance_range=destination_tolerance_range,
            max_speed=max_speed,
            goal_bonus=goal_bonus,
            max_episode_steps=max_episode_steps,
            visibility_graph=visibility_graph,
            value_tables=value_tables,
            reward_type=reward_type
        )

    def plot_observation(self, viewer, state, observation):
        pass

    def get_observation(self, state):
        # return state
        # dist_to_closest_obstacle, absolute_angle_to_closest_obstacle = self.world.range_and_bearing_to_closest_obstacle(state[0], state[1])
        obs = np.array([state[0], state[1]])
        if self.add_goal_position_to_observation:
            obs = np.concatenate([obs, self.candidate_goals['rg']])
        return obs

    def reset(self, random=False):
        return super().reset(random)
