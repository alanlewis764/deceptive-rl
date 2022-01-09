from abc import ABC, abstractmethod
from collections import defaultdict
from collections import deque

import numpy as np
import torch

from spinup.algos.pytorch.sac.sac_agent import DiscreteSacAgent, ContinuousSacAgent, SacFactory


# --------------------------------------------- OBSERVATION CLASSES ---------------------------------------------------
class Observation:
    def __init__(self, state, action) -> None:
        super().__init__()
        self.state = state
        self.action = action

    def __key(self):
        return self.state, self.action

    def __eq__(self, that: object) -> bool:
        assert isinstance(that, Observation)
        return self.__key() == that.__key()

    def __hash__(self) -> int:
        return hash(self.__key())


class ObservationTensor(Observation):
    def __init__(self, state, action) -> None:
        super().__init__(state, action)
        self.state = torch.as_tensor(state, dtype=torch.float32)
        self.action = torch.as_tensor(action, dtype=torch.float32)


# --------------------------------------------- CANDIDATE CLASSES ---------------------------------------------------
class CandidateBase(ABC):
    def __init__(self, subagent, probability: float, name="rg", q_difference_queue_length=5000) -> None:
        super().__init__()
        self.subagent = subagent
        self.probability = probability
        self.q_difference = 0  # Q-difference should change at time-step
        self.observation = None
        self.name = name  # Note: the name is super important because it needs to match the goal name

        # store the q diffs collected for each state-action of the candidates at each time-step. This will avoid the
        # need to recompute the q-difference for the selected observation when adding to the accumulated_q_diff
        self.observation_q_diff_map = defaultdict(float)

        # We also need to track the accumulated q-diff so far for the trajectory. This we mean that we don't need to
        # recompute Q-values at every step.
        self.accumulated_q_diff = 0
        # store a queue of q-differences such that old q-differences are discarded. This allows us to consider a fixed
        # number of the most recent q-differences when considering deceptive behaviour
        self.q_difference_queue = deque(maxlen=q_difference_queue_length)

    def fast_update_q_difference(self, observation: Observation, tau=1.0):
        # Rather than than recompute the q-diff for every observation in the observation sequence, we can just use
        # the stored accumulated q-diff and add the additional q-diff on. This avoids the need to do the expensive
        # recalculation of q-values over and over again.
        added_q_diff = self.observation_divergence(observation)

        # memoise the observation -> added_q_diff
        self.observation_q_diff_map[observation] = added_q_diff

        self.q_difference = (self.accumulated_q_diff + added_q_diff) / tau

        return self.q_difference

    def reset(self):
        self.accumulated_q_diff = 0
        self.q_difference_queue.clear()
        self.q_difference = 0
        self.observation = None

    def update_accumulated_q_diff(self, observation: Observation):
        # If we reach the maximum of our q, then we want to subtract the outgoing items from the accumulated q diff and
        # add the ingoing item
        if len(self.q_difference_queue) == self.q_difference_queue.maxlen:
            outgoing_q_diff = self.q_difference_queue.popleft()
            self.accumulated_q_diff -= outgoing_q_diff

        self.accumulated_q_diff += self.observation_q_diff_map[observation]
        self.q_difference_queue.append(self.observation_q_diff_map[observation])

        # reset the map back to empty for the next round so that the memory doesn't explode
        self.observation_q_diff_map = defaultdict(float)

    def observation_divergence(self, observation) -> float:
        # NOTE: this works for Tabular Q-learning since we can maximise over the action space easily (assuming a
        # discrete action space)... But it doesn't work well for high-dimensional action space...
        # What we need to do in high-dimensional action-space is:
        #   1) Estimate the value function for the state-action pair
        #   2) Select an optimal action given the state using the policy
        #   3) Determine the optimal value using the state-optimal_action pair
        #   4) Calculate the divergence
        value = self.subagent.get_value_estimate(observation.state, observation.action)
        optimal_value = self.subagent.get_max_value_estimate(observation.state)
        divergence = value - optimal_value
        return divergence

    def update_q_difference(self, observation_sequence: np.array):
        q_difference = sum(map(self.observation_divergence, observation_sequence))
        self.q_difference = q_difference
        return q_difference

    # don't update probability here because we want to only update probability when we actually select the action
    def calculate_probability(self, all_q_diffs: np.array) -> float:
        all_q_diffs = np.array(all_q_diffs)
        q_weighting = np.exp(self.q_difference)
        total_weighting = np.sum(np.exp(all_q_diffs))
        return q_weighting / total_weighting * self.probability

    def q_gain(self, observation, observation_sequence):
        q_val = self.subagent.get_value_estimate(observation.state, observation.action)
        r = self.residual_expected_reward(observation_sequence)
        return q_val - r

    def residual_expected_reward(self, observation_sequence):
        prev_obs = observation_sequence[-1]
        init_obs = observation_sequence[0]
        recent_q = self.subagent.get_value_estimate(prev_obs.state, prev_obs.action)
        init_q = self.subagent.get_value_estimate(init_obs.state, init_obs.action)
        return recent_q - init_q

    @abstractmethod
    def submit_observation(self, state, pretrained=False) -> Observation:
        raise NotImplementedError


class QLearningCandidate(CandidateBase):
    """
    A candidate class to keep track of the all the information relating to the candidate goal.
        1) The policy/subagent
        2) The probability associated with the goal
    """

    def __init__(self, subagent, probability: float, name="rg", q_difference_queue_length=5000) -> None:
        super().__init__(subagent=subagent, probability=probability, name=name,
                         q_difference_queue_length=q_difference_queue_length)

    def submit_observation(self, state, pretrained=False) -> Observation:
        """
        :param:
        :return: candidate action based on the agents policy.

        Choose an action and observation given the state and the agents policy. Update this in place
        """
        action = self.subagent.select_action(state, pretrained)
        self.observation = Observation(state, action)
        return self.observation


class PretrainedACCandidate(CandidateBase):

    def __init__(self,
                 probability,
                 state_space=None,
                 action_space=None,
                 subagent_path=None,
                 name="rg",
                 num_epochs=100,
                 q_table_path=None,
                 q_difference_queue_length=5000,
                 experiment_name=None) -> None:
        # We need to load the agent given that it is pre-trained file
        assert subagent_path is not None, "Agent must have a path to where the pretrained files are kept"
        subagent = torch.load(subagent_path)
        super().__init__(subagent=subagent,
                         probability=probability,
                         name=name,
                         q_difference_queue_length=q_difference_queue_length)

    def submit_observation(self, state, pretrained=False) -> ObservationTensor:
        """
        :param:
        :return: candidate action based on the agents policy.

        Choose an action and observation given the state and the agents policy. Update this in place
        """
        action = self.subagent.select_action(state, pretrained)
        self.observation = ObservationTensor(state, action)
        return self.observation


class OnlineCandidate(CandidateBase):
    def __init__(self,
                 probability,
                 state_space=None,
                 action_space=None,
                 subagent_path=None,
                 num_epochs=100,
                 pi_lr=1e-3,
                 critic_lr=1e-3,
                 name="rg",
                 experiment_name=None,
                 q_difference_queue_length=5000,
                 discount_rate=0.975,
                 discrete=True,
                 lr_decay=0.95,
                 alpha=0.2,
                 batch_size=128,
                 hidden_dim=64,
                 num_test_eps=1) -> None:
        if subagent_path:
            subagent = torch.load(subagent_path)
        else:
            subagent = SacFactory.create(discrete=discrete,
                                         state_space=state_space,
                                         action_space=action_space,
                                         subagent_name=name,
                                         num_epochs=num_epochs,
                                         critic_lr=critic_lr,
                                         pi_lr=pi_lr,
                                         learning_decay=lr_decay,
                                         experiment_name=experiment_name,
                                         discount_rate=discount_rate,
                                         alpha=alpha,
                                         batch_size=batch_size,
                                         hidden_dim=hidden_dim,
                                         num_test_eps=num_test_eps)

        super().__init__(subagent=subagent,
                         probability=probability,
                         name=name,
                         q_difference_queue_length=q_difference_queue_length)
        self.num_action_choices = 0

    def submit_observation(self, state, pretrained=False) -> Observation:
        action = self.subagent.select_action(state, pretrained)
        self.observation = ObservationTensor(state, action)
        return self.observation

    def add_experience(self, state, action, reward, next_state, done) -> None:
        self.subagent.add_experience(state, action, reward, next_state, done)

    def end_trajectory(self, test=False):
        self.reset()
        self.subagent.end_trajectory(test)

    def log_agent_stats(self, time_step, epoch_number):
        self.subagent.log_stats(time_step, epoch_number)

    def learn(self, time_step):
        """
        Train the individual candidates using the experiences that they have collected so far. This is the stage where
        they actually update their value estimation functions and policy function.
        This can (and should) be called at every time-step since the agent itself handles the frequency at which it
        should get updated.
        """
        self.subagent.learn(time_step=time_step)

    def save_state(self, epoch_number, train_env):
        self.subagent.save_state(epoch_number=epoch_number, train_env=train_env)

    def test_subagent(self, test_env=None, test_env_key=None):
        self.subagent.test(test_env, test_env_key)
