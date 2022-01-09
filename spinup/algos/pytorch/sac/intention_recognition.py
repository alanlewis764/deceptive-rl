from abc import ABC, abstractmethod
from collections import defaultdict

# numpy and scipy stufff
import numpy as np
import scipy.stats
from scipy.stats import entropy
from numpy.linalg import norm
import pyvisgraph as vg

# local stuff
from spinup.algos.pytorch.sac.q_table_agent import QAgent
from spinup.algos.pytorch.sac.candidate import QLearningCandidate, Observation
from gym_minigrid.env_reader import read_map_as_visibility_graph


class IntentionRecognitionBase(ABC):
    def __init__(self, num_candidates, all_model_names) -> None:
        super().__init__()
        self.num_candidates = num_candidates
        self.all_model_names = all_model_names
        self.candidate_probabilities_dict = {goal: [] for goal in self.all_model_names}
        self.current_probabilities_dict = defaultdict(float)
        self.entropies = []

    def add_probabilities(self, probabilities):
        # deceptive_values maintain the same order as the candidate list
        for candidate_name, probability in zip(self.all_model_names, probabilities):
            self.candidate_probabilities_dict[candidate_name].append(probability)
            # also maintain a dict for most recent probability access
            self.current_probabilities_dict[candidate_name] = probability

    def get_ldp(self):
        rg_probs = self.candidate_probabilities_dict['rg']
        index = len(rg_probs) - 1
        count = 0
        for i in range(index, 0, -1):
            best_fg_prob = 0
            for goal_name in self.all_model_names:
                if goal_name == 'rg':
                    continue
                fg_probs = self.candidate_probabilities_dict[goal_name]
                best_fg_prob = max(best_fg_prob, fg_probs[i])
            rg_prob = rg_probs[i]
            if rg_prob > best_fg_prob:
                count += 1
            else:
                break
        return count

    @staticmethod
    def normalise_probabilities(probabilities):
        return probabilities / norm(probabilities, ord=1)  # ord=1 for L1 norm

    @abstractmethod
    def predict_goal_probabilities(self, state, action):
        raise NotImplementedError


class DiscreteIntentionRecognition(IntentionRecognitionBase):
    def __init__(
            self,
            state_space,
            action_space,
            all_models,
            all_model_names,
            tau=1.0,
            q_difference_queue_length=5000
    ) -> None:
        super().__init__(num_candidates=len(all_models),
                         all_model_names=all_model_names)
        self.candidates = np.array(
            [QLearningCandidate(subagent=QAgent(state_space=state_space,
                                                action_space=action_space,
                                                name=name,
                                                discount_rate=0.975,
                                                learning_rate=1e-3,
                                                file_path=f),
                                probability=1.0 / self.num_candidates,
                                q_difference_queue_length=q_difference_queue_length,
                                name=name) for f, name in zip(all_models, all_model_names)], dtype=QLearningCandidate
        )
        self.tau = tau
        self.observation_sequence = []
        self.q_difference_queue_length = q_difference_queue_length

    def predict_goal_probabilities(self, state, action):
        """
        :param state: the most recently seen state
        :param action: the action taken in that state
        :return: deceptive_values over the candidate goals given by
         {
            goal_name1: goal_prob1,
            goal_name2: goal_prob2,
            ...
            goal_namek: goal_probk,
         }
        """

        observation = Observation(state=state, action=action)  # O(1)
        self.observation_sequence.append(observation)  # O(1)

        # calculate the q-difference of the observation sequence for each candidate
        q_diffs = [candidate.update_q_difference(self.observation_sequence) for candidate in
                   self.candidates]  # this is slow

        normalised_q_diffs = self.normalise_q_diffs(q_diffs)

        # calculate the deceptive_values
        probabilities = np.array([candidate.calculate_probability(all_q_diffs=normalised_q_diffs) for candidate in
                                  self.candidates])

        # normalise the deceptive_values to ensure that they sum to 1
        normalised_probabilities = self.normalise_probabilities(probabilities)

        entropy = scipy.stats.entropy(normalised_probabilities)

        self.entropies.append(entropy)

        # add the deceptive_values to the candidate-deceptive_values dictionary
        self.add_probabilities(normalised_probabilities)

        return normalised_probabilities

    def normalise_q_diffs(self, all_q_diffs):
        """
        We need to normalise and return the q diffs of all the candidates and update them in-place. We do this because
        to determine the probability we need all the q-diffs and the q-diff specific to that candidate. So each
        candidate needs to know its own q-diff specifically (hence in-place update).
        """
        normalised_q_diffs = []
        absolute_q_diffs = list(map(abs, all_q_diffs))

        qmin = min(absolute_q_diffs)
        for candidate in self.candidates:
            candidate.q_difference = - (abs(candidate.q_difference) - qmin)
            normalised_q_diffs.append(candidate.q_difference)
        return normalised_q_diffs

    def probability_of_real_goal(self):
        return self.current_probabilities_dict['rg']  # real goal name is always 'rg'


class ContinuousIntentionRecognition(IntentionRecognitionBase):

    def __init__(self, map_number, goals, start_state) -> None:
        """
        :param goals: a list of goals in the following format: ('name', x, y)
        """
        all_model_names = [goal[2] for goal in goals]
        super().__init__(num_candidates=len(goals),
                         all_model_names=all_model_names)
        self.goals = goals
        self.start_state = start_state
        self.map_number = map_number
        self.start_point = vg.Point(self.start_state[0], self.start_state[1])
        self.visibility_graph = read_map_as_visibility_graph(number=map_number)
        self.optcost_dict = {
            goal[2]: self.path_to_cost(self.visibility_graph.shortest_path(
                self.start_point,
                vg.Point(goal[0], goal[1]))
            )
            for goal in goals
        }

    def predict_goal_probabilities(self, state, action):
        cost_differences = np.array([self.cost_difference(state=state, goal=goal) for goal in self.goals])

        probabilities = np.array(
            [np.exp(cost_difference) / sum(np.exp(cost_differences)) for cost_difference in cost_differences]
        )

        normalised_probabilities = self.normalise_probabilities(probabilities)

        self.add_probabilities(normalised_probabilities)

        return normalised_probabilities

    def cost_difference(self, state, goal):
        goal_x, goal_y, goal_name = goal
        current_point = vg.Point(state[0], state[1])
        goal_point = vg.Point(goal_x, goal_y)
        optpath_given_state = self.visibility_graph.shortest_path(self.start_point, current_point) + \
                              self.visibility_graph.shortest_path(current_point, goal_point)
        optcost_given_state = self.path_to_cost(optpath_given_state)
        optcost = self.optcost_dict[goal_name]
        return optcost - optcost_given_state

    def path_to_cost(self, path):
        prev_point = path[0]
        distance = 0
        for point in path[1:]:
            distance += self.l2norm(prev_point.x, prev_point.y, point.x, point.y)
            prev_point = point
        return distance

    @staticmethod
    def l2norm(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class IntentionRecognitionFactory:
    @staticmethod
    def create(discrete, state_space, action_space, all_models, all_model_names, start_state, goals, map_num):
        if discrete:
            return DiscreteIntentionRecognition(state_space=state_space,
                                                action_space=action_space,
                                                all_models=all_models,
                                                all_model_names=all_model_names)
        else:
            return ContinuousIntentionRecognition(start_state=start_state,
                                                  goals=goals,
                                                  map_number=map_num)
