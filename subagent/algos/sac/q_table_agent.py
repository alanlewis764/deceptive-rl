import copy
import json
import numpy as np


def map_path_to_dict(fp):
    with open(fp) as f:
        my_data = json.load(f)
        # convert the keys to state tuple
        my_data = dict(map(lambda kv: ((eval(kv[0])), kv[1]), my_data.items()))
    return my_data


class Agent:
    def __init__(self, action_space, discount_rate=0.95, learning_rate=0.1, file_path=None):
        super().__init__()
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.action_space = action_space

    def select_action(self, state):
        raise NotImplementedError

    def update(self, state, action, next_state, reward, done):
        raise NotImplementedError

    def get_value_estimate(self, state, action):
        pass

    def get_max_value_estimate(self, state):
        pass

    def save(self,
             file_name,
             location='/Users/alanlewis/PycharmProjects/DeceptiveReinforcementLearning/model_storage/'):
        pass

    def load(self, file_path):
        pass


class QAgent(Agent):

    def __init__(
            self,
            state_space,
            action_space,
            name,  # needs to match up with the goal
            discount_rate=0.95,
            learning_rate=0.1,
            epsilon=0.1,
            convergence_constant=0.001,
            convergence_interval=100,
            file_path=None
    ):
        super().__init__(action_space, discount_rate, learning_rate)
        self.state_space = state_space
        self.epsilon = epsilon
        self.name = name
        self.state_space_dimension = len(self.state_space.high)
        self.action_space_dimension = self.action_space.n
        self.actions = range(action_space.n)
        if file_path is not None:
            self.Q_table = np.load(file_path)
        else:
            value_range = state_space.high.max() - state_space.low.min()
            # initialise a Q table with dimension |S| x |A|
            # Note: this has dimension num_cells X value_range X |A|
            self.Q_table = np.zeros([value_range] * self.state_space_dimension + [self.action_space_dimension])

        # store a copy of the Q-table to check for convergence periodically
        self.cached_q_table = copy.copy(self.Q_table)
        self.convergence_constant = convergence_constant
        self.convergence_interval = convergence_interval
        self.time_steps_since_convergence_check = 0

    def select_action(self, state, test=False):
        if not test and np.random.random(1) < self.epsilon:
            return self.action_space.sample()
        else:
            payoffs_per_action_given_state = self.Q_table[state]
            return np.random.choice(
                np.flatnonzero(payoffs_per_action_given_state == payoffs_per_action_given_state.max())
            )

    def update(self, state, action, next_state, reward, done):
        max_Q_next = self.get_max_value_estimate(next_state)
        self.Q_table[state][action] += self.learning_rate * (
                reward + self.discount_rate * (1 - done) * max_Q_next - self.Q_table[state][action])

    def get_value_estimate(self, state, action):
        return self.Q_table[state][action]

    def get_max_value_estimate(self, state):
        return self.Q_table[state].max()

    def estimate_value_function(self, state):
        transformed_state = tuple(state.numpy().astype(int, copy=True))
        return self.get_max_value_estimate(transformed_state)

    def has_converged(self) -> bool:
        if self.time_steps_since_convergence_check < self.convergence_interval:
            self.time_steps_since_convergence_check += 1
            return False
        else:
            converged = (self.Q_table - self.cached_q_table < self.convergence_constant).all()
            self.cached_q_table = self.Q_table.copy()
            self.time_steps_since_convergence_check = 0
            return converged

    def save(
            self,
            file_path=None,
            file_name=None,
            location='/Users/alanlewis/PycharmProjects/DeceptiveReinforcementLearning/model_storage/tabular_q/'
    ) -> None:
        if file_path:
            np.save(file_path, self.Q_table)
        else:
            np.save(location + file_name, self.Q_table)