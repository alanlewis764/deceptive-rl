# Agents
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
import copy

import torch
from scipy.special import softmax
from scipy.stats import entropy

from gym_minigrid.wrappers import *
from gym_minigrid.env_reader import read_name, get_all_model_names

from intention_recognition.intention_recognition import IntentionRecognitionBase

from subagent.algos.sac.candidate import CandidateBase, Observation, PretrainedACCandidate, OnlineCandidate
from ambiguity.ambiguity_types import AmbiguityTypes
from subagent.user_config import DEFAULT_DATA_DIR


def get_sac_path(agent_type, map_name='empty1', agent_name='rg', seed='42', discrete=True, state_visitation=False,
                 decay_param=None, pruning_constant=None, tau_decay=None, tau_constant=None):
    # find correct directory
    if map_name == 'test':
        folder = f'{DEFAULT_DATA_DIR}/test-{agent_name}/test-{agent_name}_s{seed}/'
    elif agent_type == AmbiguityTypes.SAC:
        if discrete:
            folder = f'{DEFAULT_DATA_DIR}/pretrained-sac-{map_name}-{agent_name}/pretrained-sac-{map_name}-{agent_name}_s{seed}/'
        else:
            folder = f'{DEFAULT_DATA_DIR}/continuous-pretrained-sac-{map_name}-{agent_name}/continuous-pretrained-sac-{map_name}-{agent_name}_s{seed}/'
    elif agent_type == AmbiguityTypes.ONLINE_SAC:
        if discrete:
            folder = f'{DEFAULT_DATA_DIR}/{map_name}-online-ac-softmax-{agent_name}/{map_name}-online-ac-softmax-{agent_name}_s{seed}/'
        else:
            folder = f'{DEFAULT_DATA_DIR}/continuous-{map_name}-online-ac-softmax-{agent_name}/continuous-{map_name}-online-ac-softmax-{agent_name}_s{seed}/'
    elif agent_type == AmbiguityTypes.INTERVAL_SAC:
        if discrete:
            folder = f'{DEFAULT_DATA_DIR}/interval_{map_name}-{agent_name}/interval_{map_name}-{agent_name}_s{seed}/'
        else:
            folder = f'{DEFAULT_DATA_DIR}/continuous-interval_{map_name}-{agent_name}/continuous-interval_{map_name}-{agent_name}_s{seed}/'
    elif agent_type == AmbiguityTypes.INTERVAL_ONLINE_SAC:
        if discrete:
            folder = f'{DEFAULT_DATA_DIR}/interval_{map_name}-online-ac-softmax-{agent_name}/interval_{map_name}-online-ac-softmax-{agent_name}_s{seed}/'
        else:
            folder = f'{DEFAULT_DATA_DIR}/continuous-interval_{map_name}-online-ac-softmax-{agent_name}/continuous-interval_{map_name}-online-ac-softmax-{agent_name}_s{seed}/'
    elif agent_type == AmbiguityTypes.PRUNING_DECAY:
        if discrete:
            folder = f'{DEFAULT_DATA_DIR}/{map_name}-online-ac-softmax-pruning-decay={decay_param}-{agent_name}/{map_name}-online-ac-softmax-pruning-decay={decay_param}-{agent_name}_s{seed}/'
        else:
            folder = f'{DEFAULT_DATA_DIR}/continuous-{map_name}-online-ac-softmax-pruning-decay={decay_param}-{agent_name}/continuous-{map_name}-online-ac-softmax-pruning-decay={decay_param}-{agent_name}_s{seed}/'
    elif agent_type == AmbiguityTypes.PRUNING_CONSTANT:
        if discrete:
            folder = f'{DEFAULT_DATA_DIR}/{map_name}-online-ac-softmax-pruning-constant={pruning_constant}-{agent_name}/{map_name}-online-ac-softmax-pruning-constant={pruning_constant}-{agent_name}_s{seed}/'
        else:
            folder = f'{DEFAULT_DATA_DIR}/continuous-{map_name}-online-ac-softmax-pruning-constant={pruning_constant}-{agent_name}/continuous-{map_name}-online-ac-softmax-pruning-constant={pruning_constant}-{agent_name}_s{seed}/'
    elif agent_type == AmbiguityTypes.TAU_DECAY:
        if discrete:
            folder = f'{DEFAULT_DATA_DIR}/{map_name}-online-ac-softmax-tau-decay={tau_decay}-{agent_name}/{map_name}-online-ac-softmax-tau-decay={tau_decay}-{agent_name}_s{seed}/'
        else:
            folder = f'{DEFAULT_DATA_DIR}/continuous-{map_name}-online-ac-softmax-tau-decay={tau_decay}-{agent_name}/continuous-{map_name}-online-ac-softmax-tau-decay={tau_decay}-{agent_name}_s{seed}/'
    elif agent_type == AmbiguityTypes.TAU_CONSTANT:
        if tau_constant == 0:
            folder = f'{DEFAULT_DATA_DIR}/{map_name}-online-ac-hardmax-{agent_name}/{map_name}-online-ac-hardmax-{agent_name}_s{seed}/'
        else:
            if discrete:
                folder = f'{DEFAULT_DATA_DIR}/{map_name}-online-ac-softmax-tau-constant={tau_constant}-{agent_name}/{map_name}-online-ac-softmax-tau-constant={tau_constant}-{agent_name}_s{seed}/'
            else:
                folder = f'{DEFAULT_DATA_DIR}/continuous-{map_name}-online-ac-softmax-tau-constant={tau_constant}-{agent_name}/continuous-{map_name}-online-ac-softmax-tau-constant={tau_constant}-{agent_name}_s{seed}/'
    else:
        raise ValueError(f"{agent_type} is not a valid type")

    if state_visitation:
        return folder + 'train_state_visitation_dict.json'
    else:
        # return model by default
        return folder + 'pyt_save/model.pt'


ACTION_TO_PATH_COST = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: np.sqrt(2),
    5: np.sqrt(2),
    6: np.sqrt(2),
    7: np.sqrt(2),
    8: 0,
}


class AmbiguityBase(ABC):
    def __init__(self,
                 state_space,
                 action_space,
                 name,
                 all_models,
                 real_goal_pruning_constant=0.0,
                 q_gain_pruning_constant=0.0,
                 experiment_name='ambiguity-base') -> None:
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.name = name
        self.num_subagents = len(all_models)

        # attributes to be overridden
        self.candidates = None
        self.pruned_candidates = None
        self.real_pretrained_subagent = None

        # information store needed for deception
        self.observation_sequence = []  # dynamic data-structure so numpy arrays aren't really suitable
        self.all_q_differences = None
        self.current_state_value_estimate = -np.inf

        # hyper-params
        self.real_goal_pruning_constant = real_goal_pruning_constant
        self.q_gain_pruning_constant = q_gain_pruning_constant

    # This works correctly
    def entropy_of_candidate_observation(self, candidate_observation: Observation) -> float:
        """
        :param candidate_observation: the observation submitted by one of the candidate sub-agents
        :return: the entropy of the over the probability of the candidate goals for that submitted observation
        """
        # assume that the observation is included as part of the observation sequence. Do not do this in-place since it
        # has not yet been chosen as the observation.
        candidate_observation_sequence = np.append(self.observation_sequence, [candidate_observation])

        # remove the candidates that are obviously irrational given the submitted observation from the entropy
        # calculation
        # rational_candidates = self.prune_irrational_candidates(candidate_observation)
        rational_candidates = self.candidates

        # calculate the q differences of all the candidate reward functions if this is the submitted observation.
        # this also updates the q-difference for each candidate in-place
        # all_q_diffs = [candidate.update_q_difference(candidate_observation_sequence) for candidate in self.candidates]
        all_q_diffs = [candidate.fast_update_q_difference(candidate_observation) for candidate in self.candidates]

        # we need to normalise the Q-difference in case it blows up... For instance, it runs for a long time without
        # reaching the one of the goals, the Q-difference becomes pretty large
        normalised_q_diffs = self.normalise_q_diffs(all_q_diffs)

        # calculate the deceptive_values of each goal given the q-differences
        probabilities = [candidate.calculate_probability(all_q_diffs=normalised_q_diffs) for candidate in
                         rational_candidates]

        # return the entropy of the deceptive_values. This handles the normalisation and returns a positive entropy value
        return entropy(probabilities)

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

    def action_that_maximises_entropy(self):
        # if there are no candidates that make progression to the true goal (this shouldn't happen since the true
        # candidate should progress to the true gaol always) make all the candidates part of the pruned candidate set
        if not self.pruned_candidates:
            # move toward the true goal
            return self.candidates[0].observation.action
            # self.pruned_candidates = self.candidates
        entropies = np.array(
            [self.entropy_of_candidate_observation(candidate.observation) for candidate in self.pruned_candidates]
        )
        assert not np.isnan(entropies).any(), "Entropy cannot be NaN"
        choice = np.random.choice(np.flatnonzero(entropies == entropies.max()))
        selected_observation = self.pruned_candidates[choice].observation
        self.update_accumulated_q_diffs(selected_observation)
        return selected_observation.action

    def act(self, state):
        # choose actions for each candidate agents and update their candidate observations
        candidate_observations = self.select_candidate_observations(state=state)

        # prune the actions that do not progress to the real goal
        pruned_candidates = self.prune_backward_actions(state)

        # select the action that maximises the entropy of the observation sequence
        selected_action = self.action_that_maximises_entropy()

        return selected_action

    def update(self, state, action, next_state, reward, done):
        # add the selected observation to the observation sequence
        selected_observation = Observation(state=state, action=action)
        self.observation_sequence.append(selected_observation)

        # re-estimate the value of the current state for pruning
        # note, we need to use the previous (state, action) because we can't use the current state since we do not know
        # the action... If we choose the maximum value estimate then this will result in all actions being pruned except
        # the optimal action...
        self.current_state_value_estimate = self.real_pretrained_subagent.get_value_estimate(state, action)

    def update_accumulated_q_diffs(self, selected_observation):
        for candidate in self.candidates:
            candidate.update_accumulated_q_diff(selected_observation)

    def prune_backward_actions(self, state) -> typing.List[CandidateBase]:
        """
        This prunes actions submitted by the candidates which move away from the true goal.
        """
        pruned_candidates = list(
            filter(lambda candidate: self.q_gain_pruning_function(candidate=candidate, state=state),
                   self.candidates))
        self.pruned_candidates = pruned_candidates
        # if all candidates are pruned, then select the action that moves towards the real goal
        if len(pruned_candidates) == 0:
            self.pruned_candidates = [self.candidates[0]]
        return self.pruned_candidates

    def prune_irrational_candidates(self, observation):
        """
        This prunes candidates from the entropy calculation is they are obviously irrational given the submitted
        observation. This prevents them from distorting the entropy calculation.
        """
        # if there are no observations yet then don't do any pruning
        if len(self.observation_sequence) == 0:
            return self.pruned_candidates
        # We only need to consider the candidates remaining in the pruned candidate set (A*).
        final_candidates = []
        for candidate in self.pruned_candidates:
            if candidate.q_gain(observation, self.observation_sequence) >= self.q_gain_pruning_constant:
                final_candidates.append(candidate)
        return final_candidates

    @abstractmethod
    def select_candidate_observations(self, state) -> typing.List[Observation]:
        raise NotImplementedError

    @abstractmethod
    def q_gain_pruning_function(self, candidate: CandidateBase, state) -> bool:
        raise NotImplementedError


class ACAmbiguityAgent(AmbiguityBase):

    def __init__(self, state_space, action_space, name, all_models, all_model_names, rg_model,
                 real_goal_pruning_constant=0.0, q_gain_pruning_constant=0.0,
                 experiment_name='ac-ambiguity-agent', discrete=True) -> None:
        super().__init__(state_space, action_space, name, all_models, real_goal_pruning_constant,
                         q_gain_pruning_constant, experiment_name)
        self.candidates = np.array([
            PretrainedACCandidate(probability=1.0 / self.num_subagents,
                                  state_space=state_space,
                                  action_space=action_space,
                                  subagent_path=model_path,
                                  name=name,
                                  experiment_name=experiment_name)
            for model_path, name in zip(all_models, all_model_names)
        ], dtype=PretrainedACCandidate)
        # This doesn't need to know the goal that it is heading towards since it a pre-trained
        self.real_pretrained_subagent = torch.load(rg_model)
        self.discrete = discrete

    def act(self, state):
        selected_action = super().act(state)

        # need to convert this into an int since it is a tensor
        if self.discrete:
            selected_action = selected_action.int()
        else:
            selected_action = selected_action

        return selected_action

    def select_candidate_observations(self, state) -> typing.List[Observation]:
        # This updates the candidate inplace to include its selected observation given the state
        selected_observations = [candidate.submit_observation(state, pretrained=True) for candidate in
                                 self.candidates]
        return selected_observations

    def q_gain_pruning_function(self, candidate: CandidateBase, state) -> bool:
        action = candidate.observation.action
        action_value = self.real_pretrained_subagent.get_value_estimate(state, action)
        current_value = self.current_state_value_estimate
        return action_value - current_value > self.real_goal_pruning_constant

    def single_environment_run(self, env_detail, agent_type='interval_sac', decay_param=None, pruning_constant=None,
                               tau_decay=None, tau_constant=None, measure='real_goal_probs', discrete=True,
                               intention_recognition: IntentionRecognitionBase = None, render=False):
        """
        This should be a single environment run to be used after training for analysis of the results
        :return:
        """
        env, env_name = env_detail
        state_visitation_dict = defaultdict(int)
        state = env.reset()
        done = False
        path_cost = 0
        num_steps = 0
        max_steps = 1000

        while not done and num_steps < max_steps:
            if render:
                env.render()
            state_visitation_dict[str((state[0], state[1]))] += 1
            tensor_state = torch.as_tensor(state, dtype=torch.float32)
            action = self.act(tensor_state)
            if discrete:
                path_cost += ACTION_TO_PATH_COST[int(action)]
            else:
                path_cost += 1

            if intention_recognition is not None:
                _ = intention_recognition.predict_goal_probabilities(state, action)

            next_state, reward, done, info = env.step(action)
            self.update(tensor_state, action, next_state, reward, done)
            state = next_state
            num_steps += 1
        env.close()
        state_visitation_dict[str((state[0], state[1]))] += 1
        if intention_recognition is not None:
            if measure == 'real_goal_probs':
                score = intention_recognition.candidate_probabilities_dict['rg']
            elif measure == 'entropy':
                score = intention_recognition.entropies
            elif measure == 'ldp':
                score = intention_recognition.get_ldp()
            else:
                raise ValueError("Not a valid intention recognition score")
            return score, path_cost
        else:
            return 'NO SCORE', 'NO_PATH_COST'


class OnlineACAmbiguityAgent(ACAmbiguityAgent):

    def __init__(self, state_space, action_space, name, all_models, all_model_names, env,
                 real_goal_pruning_constant=0, q_gain_pruning_constant=0, num_epochs=100, start_steps=40000,
                 max_ep_len=49 ** 2, steps_per_epoch=16000, pi_lr=1e-3, critic_lr=1e-3, tau=1, tau_decay=0.99,
                 policy='softmax', experiment_name='online-ac-ambiguity-agent', adaptive_pruning_constant=-20,
                 pruning_decay=0.975, discount_rate=0.975, discrete=True, lr_decay=0.95, alpha=0.2,
                 batch_size=128, hidden_dim=64, num_test_eps=1) -> None:
        AmbiguityBase.__init__(
            self, state_space=state_space, action_space=action_space, name=name, all_models=all_models,
            real_goal_pruning_constant=real_goal_pruning_constant, q_gain_pruning_constant=q_gain_pruning_constant,
            experiment_name=experiment_name)
        alpha = 0.2 if discrete else 0.01
        self.candidates = np.array([
            OnlineCandidate(probability=1.0 / self.num_subagents,
                            state_space=state_space,
                            action_space=action_space,
                            subagent_path=model_path,
                            name=name,
                            num_epochs=num_epochs,
                            experiment_name=experiment_name,
                            critic_lr=critic_lr,
                            pi_lr=pi_lr,
                            discount_rate=discount_rate,
                            discrete=discrete,
                            lr_decay=lr_decay,
                            alpha=alpha,
                            batch_size=batch_size,
                            hidden_dim=hidden_dim,
                            num_test_eps=num_test_eps)
            for model_path, name in zip(all_models, all_model_names)
        ], dtype=OnlineCandidate)
        self.real_pretrained_subagent = self.candidates[0].subagent
        self.env = env
        self.candidate_selection_dict = defaultdict(int)
        self.action_selected_dict = defaultdict(int)

        # params set for controlling training behaviour
        self.num_epochs = num_epochs
        self.start_steps = start_steps
        self.max_ep_len = max_ep_len
        self.steps_per_epoch = steps_per_epoch
        self.current_candidates_turn = 0
        self.total_steps_taken = 0
        self.discrete = discrete

        # params to control softmax of the policy
        self.policy = policy
        self.tau = tau
        self.tau_decay = tau_decay
        self.adaptive_pruning_constant = adaptive_pruning_constant
        self.pruning_decay = pruning_decay

    def entropy_of_candidate_observation(self, candidate_observation: Observation) -> float:
        """
        :param candidate_observation: the observation submitted by one of the candidate sub-agents
        :return: the entropy of the over the probability of the candidate goals for that submitted observation
        """

        # assume that the observation is included as part of the observation sequence. Do not do this in-place since it
        # has not yet been chosen as the observation.
        candidate_observation_sequence = np.append(self.observation_sequence, [candidate_observation])

        # calculate the q differences of all the candidate reward functions if this is the submitted observation.
        # this also updates the q-difference for each candidate in-place.
        all_q_diffs = np.array([candidate.fast_update_q_difference(candidate_observation) for candidate in
                                self.candidates])

        # we need to normalise the Q-difference in case it blows up... For instance, it it runs for a long time without
        # reaching the one of the goals, the Q-difference becomes pretty large
        normalised_q_diffs = self.normalise_q_diffs(all_q_diffs)

        # calculate the deceptive_values of each goal given the q-differences (this is not normalised)
        probabilities = [candidate.calculate_probability(all_q_diffs=normalised_q_diffs) for candidate in
                         self.candidates]

        # return the entropy of the deceptive_values. This handles the normalisation and returns a positve entropy value
        return entropy(probabilities)

    def select_candidate_observations(self, state) -> typing.List[Observation]:
        # This updates the candidate inplace to include its selected observation given the state
        selected_observations = [candidate.submit_observation(state, pretrained=False) for candidate in
                                 self.candidates]
        return selected_observations

    def q_gain_pruning_function(self, candidate: CandidateBase, state) -> bool:
        candidate_name = candidate.name
        action = candidate.observation.action
        action_value = self.real_pretrained_subagent.get_value_estimate(state, action)
        current_value = self.current_state_value_estimate
        return action_value - current_value > self.adaptive_pruning_constant

    def update(self, state, action, reward, next_state, done, training=True):
        # Split out the reward such that it is individual to each agent and add the experience in the agent such that it
        # can use it to learn...
        if training:
            for candidate in self.candidates:
                candidate_reward = reward[candidate.name]
                candidate.add_experience(state, action, candidate_reward, next_state, done)

        # Do all the necessary stuff that is needed to update the observation sequence and deceptive_values
        return super().update(state, action, next_state, reward, done)

    def candidate_entropies(self):
        if len(self.pruned_candidates) == 0:
            # account for the case in which there are no candidate actions that make progress to the real goal. This is
            # unlikely, but possible since the candidates actions are stochastic.
            relevant_candidates = [self.candidates[0]]  # choose the action selected by the real agent in this case
        else:
            relevant_candidates = self.pruned_candidates
        entropies = np.array(
            [self.entropy_of_candidate_observation(candidate.observation) for candidate in relevant_candidates]
        )
        return entropies

    def act(self, state):
        """
        Determine actions using the policy of the ambiguity agent. However, unlike the standard ambiguity agent, prune
        actions with adaptive pruning constant. This is because we want to learn behaviour toward all goals, not just
        the true goal, which is what happens when we prune.
        """

        # choose actions for each candidate agents and update their candidate observations
        candidate_observations = self.select_candidate_observations(state=state)

        # prune using adaptive pruning constant
        pruned_candidates = self.prune_backward_actions(state)

        # get entropies from the observation
        entropies = self.candidate_entropies()

        # select the action that maximises the entropy of the observation sequence
        if self.policy == 'softmax':
            selected_observation = self.observation_that_soft_maximises_entropy(entropies=entropies)
        elif self.policy == 'round_robin':
            selected_observation = self.observation_round_robin()
        elif self.policy == 'epoch_round_robin':
            selected_observation = self.epoch_round_robin()
        else:
            # hard max by default
            selected_observation = self.observation_that_maximises_entropy(entropies=entropies)

        # ensure that each candidate adds this to their accumulated q-diff for the observation sequence
        self.update_accumulated_q_diffs(selected_observation)

        # extract the action from the selected observation
        if self.discrete:
            selected_action = selected_observation.action.int()
        else:
            selected_action = selected_observation.action

        # self.action_selected_dict[int(selected_action)] += 1

        return selected_action

    # ------------------------------------------------- POLICIES -----------------------------------------------------#
    def observation_that_maximises_entropy(self, entropies):
        assert not np.isnan(entropies).any(), "Entropy cannot be NaN"
        choice = np.random.choice(
            np.flatnonzero(entropies == entropies.max())
        )
        self.candidate_selection_dict[choice] += 1
        return self.candidates[choice].observation

    def observation_that_soft_maximises_entropy(self, entropies):
        assert not np.isnan(entropies).any(), "Entropy cannot be NaN"
        # tau controls the temperature of softmax.
        # as tau -> 0, softmax -> hardmax
        softmax_entropies = softmax(entropies / self.tau)
        softmax_entropies = np.asarray(softmax_entropies).astype('float64')
        softmax_entropies /= softmax_entropies.sum()
        if np.fabs(softmax_entropies.sum() - 1) > np.finfo(np.float64).eps:
            print("hello")
        choice = np.random.choice(self.pruned_candidates, p=softmax_entropies)
        choice.num_action_choices += 1
        tau = self.tau
        action_choices = [candidate.observation.action for candidate in self.candidates]
        return choice.observation

    def observation_round_robin(self):
        selected_candidate = self.candidates[self.current_candidates_turn]
        self.current_candidates_turn = (self.current_candidates_turn + 1) % self.num_subagents
        return selected_candidate.observation

    def epoch_round_robin(self):
        selected_candidate = self.candidates[self.current_candidates_turn]
        return selected_candidate.observation

    # ----------------------------------------------------------------------------------------------------------------#

    def update_accumulated_q_diffs(self, selected_observation):
        for candidate in self.candidates:
            candidate.update_accumulated_q_diff(selected_observation)

    def learn(self, time_step) -> None:
        """
        Train the individual candidates using the experiences that they have collected so far. This is the stage where
        they actually update their value estimation functions and policy function.
        This can (and should) be called at every time-step since the agent itself handles the frequency at which it
        should get updated.
        """
        for candidate in self.candidates:
            candidate.learn(time_step)

    def train(self, my_train_env):
        """
        Defines the entire training loop for the agent. This should have multiple epochs, time-steps etc
        :return:
        """
        state = my_train_env.reset()
        total_steps = self.steps_per_epoch * self.num_epochs
        episode_length = 0

        for time_step in range(total_steps):
            # my_train_env.render()
            # act randomly at the start to for better exploration
            if time_step < self.start_steps:
                action = my_train_env.action_space.sample()
            else:
                action = self.act(state)

            next_state, reward, done, _ = my_train_env.step(action)
            done = False if episode_length == self.max_ep_len else done
            self.update(state, action, reward, next_state, done)
            episode_length += 1
            state = next_state

            if done or episode_length == self.max_ep_len:
                self.end_trajectory()
                state = my_train_env.reset()
                # reset the information that is specific to the run of the environment
                self.reset()
                episode_length = 0

            self.learn(time_step)

            # handle the end of an epoch
            if (time_step + 1) % self.steps_per_epoch == 0:
                epoch_number = (time_step + 1) // self.steps_per_epoch

                # 1) save the model
                self.save_agent_state(epoch_number=epoch_number, my_train_env=my_train_env)

                self.end_trajectory(test=False)

                # 2) test the agent
                self.test_candidates(test_env=self.env)

                # 3) log the statistics of the agents
                self.log_agent_stats(time_step, epoch_number)

                # degrade the softmax temperature (tau) and adaptive pruning constant
                if self.policy == 'softmax':
                    self.tau = self.tau * self.tau_decay
                self.adaptive_pruning_constant = self.adaptive_pruning_constant * self.pruning_decay
                if self.policy == 'epoch_round_robin':
                    self.current_candidates_turn = (self.current_candidates_turn + 1) % self.num_subagents

    def interval_train(self, my_train_env, num_steps):
        state = my_train_env.reset()
        episode_length = 0
        for _ in range(num_steps):
            # act randomly at the start to for better exploration
            if self.total_steps_taken < self.start_steps:
                action = my_train_env.action_space.sample()
            else:
                action = self.act(state)

            next_state, reward, done, _ = my_train_env.step(action)
            self.update(state, action, reward, next_state, done)
            state = next_state
            done = False if episode_length == self.max_ep_len else done

            if done or episode_length == self.max_ep_len:
                episode_length = 0
                self.end_trajectory()
                state = my_train_env.reset()
                # reset the information that is specific to the run of the environment
                self.reset()

            self.learn(self.total_steps_taken)

            # handle the end of an epoch
            if (self.total_steps_taken + 1) % self.steps_per_epoch == 0:
                epoch_number = (self.total_steps_taken + 1) // self.steps_per_epoch

                # 1) save the model
                self.save_agent_state(epoch_number=epoch_number, my_train_env=my_train_env)

                # 2) test the agent
                self.test_candidates(test_env=self.env)

                # 3) log the statistics of the agents
                self.log_agent_stats(self.total_steps_taken, epoch_number)

                # degrade the softmax temperature (tau)
                if self.policy == 'softmax':
                    self.tau = self.tau * self.tau_decay
                self.adaptive_pruning_constant = self.adaptive_pruning_constant * self.pruning_decay

            episode_length += 1
            self.total_steps_taken += 1

    def end_trajectory(self, test=False):
        for candidate in self.candidates:
            candidate.end_trajectory(test)

    def test_candidates(self, test_env=None, test_env_key=None):
        for candidate in self.candidates:
            candidate.test_subagent(test_env, test_env_key)

    def reset(self):
        self.observation_sequence.clear()
        self.current_state_value_estimate = -np.inf

    def log_agent_stats(self, time_step, epoch_number):
        for candidate in self.candidates:
            candidate.log_agent_stats(time_step=time_step, epoch_number=epoch_number)
            print(f'agent {candidate.name} total actions chosen = {candidate.num_action_choices}')

    def save_agent_state(self, epoch_number, my_train_env):
        """
        We need to save the state of the all the individual sub-agents
        """
        for candidate in self.candidates:
            candidate.save_state(epoch_number=epoch_number, train_env=my_train_env)

    def single_environment_run(self, env_detail, agent_type='interval_sac', decay_param=None, pruning_constant=None,
                               tau_decay=None, tau_constant=None, measure='real_goal_probs', discrete=True,
                               intention_recognition: IntentionRecognitionBase = None, render=False):
        """
        This should be a single environment run to be used after training for analysis of the results
        :return:
        """
        # make sure that you end the trajectory and reset the environment before doing this test run, otherwise
        # information from the previous run will corrupt the behaviour
        # self.end_trajectory()
        self.reset()
        env, env_name = env_detail
        env = copy.deepcopy(env)
        state = env.reset()
        done = False
        path_cost = 0

        while not done:
            tensor_state = torch.as_tensor(state, dtype=torch.float32)
            action = self.act(tensor_state)
            path_cost += ACTION_TO_PATH_COST[int(action)]
            if intention_recognition is not None:
                _ = intention_recognition.predict_goal_probabilities(state, action)
            next_state, reward, done, info = env.step(action)
            self.update(state, action, next_state, reward, done, training=False)
            state = next_state

        if intention_recognition is not None:
            if measure == 'real_goal_probs':
                score = intention_recognition.candidate_probabilities_dict['rg']
            elif measure == 'entropy':
                score = intention_recognition.entropies
            else:
                raise ValueError("Not a valid intention recognition score")
            return score, path_cost
        else:
            return 'NO SCORE', 'NO_PATH_COST'


class AmbiguityFactory:

    @staticmethod
    def create(state_space, action_space, agent_type, env_number, discrete=True, pruning_decay=None,
               pruning_constant=None, tau_decay=None, tau_constant=None):
        map_name = read_name(number=env_number, discrete=discrete)
        agent_names = get_all_model_names(env_number)
        return ACAmbiguityAgent(
            state_space=state_space,
            action_space=action_space,
            name='AC',
            all_models=[
                get_sac_path(agent_type=agent_type, map_name=f'{map_name}{env_number}', agent_name=name,
                             pruning_constant=pruning_constant, decay_param=pruning_decay,
                             tau_decay=tau_decay, tau_constant=tau_constant, discrete=discrete)
                for name in agent_names],
            rg_model=get_sac_path(agent_type=agent_type, map_name=f'{map_name}{env_number}',
                                  agent_name='rg', pruning_constant=pruning_constant,
                                  decay_param=pruning_decay, tau_decay=tau_decay,
                                  tau_constant=tau_constant, discrete=discrete),
            all_model_names=agent_names,
            real_goal_pruning_constant=pruning_constant * (pruning_decay ** 120),
            q_gain_pruning_constant=pruning_constant * (pruning_decay ** 120),
            discrete=discrete
        )
