import multiprocessing as mp
import argparse

import numpy as np
import torch
from collections import defaultdict
from gym_minigrid.env_reader import read_map, read_grid_size, get_all_model_names, read_start, read_goals, \
    get_all_models
from intention_recognition.intention_recognition import IntentionRecognitionFactory
from ambiguity.ambiguity import OnlineACAmbiguityAgent, AmbiguityFactory
from ambiguity.ambiguity_types import AmbiguityTypes
from subagent.algos.sac.sac_agent import SacFactory, SacBaseAgent
from subagent.algos.sac.data_parsing import convert_to_time_density, append_results_to_json, \
    get_interval_path_cost_json_path, get_interval_accumulated_deceptiveness_json_path


def run_online_ac_ambiguity(num_env, policy_type='softmax', discrete=True, adaptive_pruning_constant=-100,
                            pruning_decay=0.95, tau_constant=1, tau_decay=0.975, alpha=0.2, reward_type='path_cost',
                            hyper_param_study=None):
    the_train_env, map_name = read_map(num_env, random_start=False, discrete=discrete, reward_type=reward_type)
    the_test_env, map_name = read_map(num_env, random_start=False, discrete=discrete, reward_type=reward_type)

    if hyper_param_study == 'pruning_constant':
        experiment_name = f'{map_name}{num_env}-online-ac-{policy_type}-pruning-constant={adaptive_pruning_constant}'
    elif hyper_param_study == 'tau_constant':
        experiment_name = f'{map_name}{num_env}-online-ac-{policy_type}-tau-constant={tau_constant}'
    elif hyper_param_study == 'tau_decay':
        experiment_name = f'{map_name}{num_env}-online-ac-{policy_type}-tau-decay={tau_decay}'
    elif hyper_param_study == 'pruning_decay':
        experiment_name = f'{map_name}{num_env}-online-ac-{policy_type}-pruning-decay={pruning_decay}'
    else:
        experiment_name = f'{map_name}{num_env}-online-ac-{policy_type}'
    if not discrete:
        experiment_name += '-continuous'

    if num_env in {1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 25, 26, 27, 28, 33, 34, 35, 36}:
        all_models = [None, None, None]
        all_model_names = ['rg', 'fg1', 'fg2']
    else:
        all_models = [None, None, None, None, None]
        all_model_names = ['rg', 'fg1', 'fg2', 'fg3', 'fg4']

    max_ep_length = 49 ** 2 if num_env < 33 else 100 ** 2

    agent = OnlineACAmbiguityAgent(
        state_space=the_train_env.observation_space,
        action_space=the_train_env.action_space,
        name=f'OnlineACContinuous',
        all_models=all_models,
        all_model_names=all_model_names,
        env=the_test_env,
        real_goal_pruning_constant=0,
        policy=policy_type,
        adaptive_pruning_constant=adaptive_pruning_constant,
        pruning_decay=pruning_decay,
        experiment_name=experiment_name,
        discrete=discrete,
        max_ep_len=max_ep_length,
        alpha=alpha,
        tau=tau_constant,
        tau_decay=tau_decay,
        steps_per_epoch=max_ep_length * 2,
        start_steps=max_ep_length * 8,
        batch_size=128,
        hidden_dim=64,
        discount_rate=0.975,
        lr_decay=0.95,
        q_gain_pruning_constant=0,
        critic_lr=1e-3,
        pi_lr=1e-3,
        num_epochs=120,
        num_test_eps=1,
    )

    agent.train(the_train_env)


def train_online_ambiguity_vs_pruning_decay(map_num, discrete=True, alpha=0.2, reward_type='path_cost'):
    policy = 'softmax'
    decay_params = [1, 0.975, 0.95, 0.9, 0.75, 0.5]
    adaptive_pruning_constant = -100
    tau_constant = 1.0
    tau_decay = 0.975
    pool = mp.Pool(len(decay_params))
    pool.starmap(run_online_ac_ambiguity,
                 [(map_num, policy, discrete, adaptive_pruning_constant, decay_param, tau_constant, tau_decay,
                   alpha, reward_type, 'pruning_decay') for decay_param in decay_params])


def train_online_ambiguity_vs_pruning_constant(map_num, discrete=True, alpha=0.2, reward_type='path_cost'):
    policy = 'softmax'
    pruning_constants = [-1, -10, -50, -100, -500, -1000]
    decay_param = 0.95
    tau_constant = 1.0
    tau_decay = 0.975
    pool = mp.Pool(len(pruning_constants))
    pool.starmap(run_online_ac_ambiguity,
                 [(map_num, policy, discrete, pruning_constant, decay_param, tau_constant, tau_decay, alpha,
                   reward_type, 'pruning_constant') for pruning_constant in pruning_constants])


def train_online_ambiguity_vs_tau_constant(map_num, discrete=True, alpha=0.2, reward_type='path_cost'):
    policy = 'softmax'
    tau_constants = [0, 0.5, 1.0, 2.0, 5.0, 10.0]
    tau_decay = 0.975
    decay_param = 0.90
    adaptive_pruning_constant = 0
    pool = mp.Pool(len(tau_constants))
    pool.starmap(run_online_ac_ambiguity,
                 [(map_num, policy, discrete, adaptive_pruning_constant, decay_param, tau_constant, tau_decay, alpha,
                   reward_type, 'tau_constant') for tau_constant in tau_constants])


def train_online_ambiguity_vs_tau_decay(map_num, discrete=True, alpha=0.2, reward_type='path_cost'):
    policy = 'softmax'
    tau_constant = 5.0
    tau_decays = [1, 0.99, 0.95, 0.9, 0.75, 0.5]
    decay_param = 0.90
    adaptive_pruning_constant = -10
    pool = mp.Pool(len(tau_decays))
    pool.starmap(run_online_ac_ambiguity,
                 [(map_num, policy, discrete, adaptive_pruning_constant, decay_param, tau_constant, tau_decay, alpha,
                   reward_type, 'tau_decay') for tau_decay in tau_decays])


def online_agent_vs_training_cost(map_number, adaptive_pruning_constant=-100, pruning_decay=0.95, discrete=True,
                                  reward_type='value_table', num_intervals=10, alpha=0.2):
    size = read_grid_size(number=map_number)[0]
    train_env, map_name = read_map(number=map_number, random_start=False, terminate_at_any_goal=False, goal_name='rg',
                                   discrete=discrete, max_episode_steps=size ** 2, reward_type=reward_type)
    test_env, map_name = read_map(number=map_number, random_start=False, terminate_at_any_goal=False, goal_name='rg',
                                  discrete=discrete, max_episode_steps=size ** 2, destination_tolerance_range=2,
                                  reward_type=reward_type)

    all_models = get_all_models(map_num=map_number)
    all_model_names = get_all_model_names(map_num=map_number)

    policy_type = 'softmax'
    experiment_name = f'interval_{map_name}{map_number}-online-ac-{policy_type}' if discrete else f'continuous-interval_{map_name}{map_number}-online-ac-{policy_type}'
    agent = OnlineACAmbiguityAgent(
        state_space=train_env.observation_space,
        action_space=train_env.action_space,
        name='IntervalOnlineAC',
        all_models=all_models,
        all_model_names=all_model_names,
        env=test_env,
        max_ep_len=size ** 2,
        policy=policy_type,
        experiment_name=experiment_name,
        adaptive_pruning_constant=adaptive_pruning_constant,
        pruning_decay=pruning_decay,
        discrete=discrete,
        alpha=alpha,
        start_steps=(size ** 2) * 2 * 4,
        steps_per_epoch=(size ** 2) * 2,
        num_epochs=200,
        critic_lr=3e-4,
        pi_lr=3e-4,
        lr_decay=1,
        tau=1,
        tau_decay=0.975,
        discount_rate=0.975,
        real_goal_pruning_constant=0,
        q_gain_pruning_constant=0
    )

    path_costs_vs_train_step = defaultdict(float)
    accumulated_deceptiveness_vs_train_step = defaultdict(float)
    training_interval_steps = 120000 if len(all_model_names) == 3 else 150000
    for i in range(num_intervals):
        # do a round of training
        agent.interval_train(my_train_env=train_env, num_steps=training_interval_steps)

        # test the agent
        rg_probs, path_cost = test_interval(map_number, agent_type=AmbiguityTypes.INTERVAL_ONLINE_SAC,
                                            model_config='interval_online_sac', discrete=discrete)

        # measure real goal probs at different stages to avoid high deceptiveness from agents that take lots of steps
        rg_probs = convert_to_time_density(rg_probs).values()
        accumulated_deceptiveness = sum(map(lambda p: 1 - p, rg_probs))
        print(f"accumulated = {accumulated_deceptiveness}")
        print(f"path cost = {path_cost}")
        path_costs_vs_train_step[(training_interval_steps / 10) * (i + 1)] = path_cost
        accumulated_deceptiveness_vs_train_step[(training_interval_steps / 10) * (i + 1)] = accumulated_deceptiveness

    append_results_to_json(
        fp=get_interval_accumulated_deceptiveness_json_path('interval_online_sac',
                                                            discrete=discrete),
        key=f'{map_name}{map_number}',
        results=accumulated_deceptiveness_vs_train_step
    )
    append_results_to_json(
        fp=get_interval_path_cost_json_path('interval_online_sac', discrete=discrete),
        key=f'{map_name}{map_number}',
        results=path_costs_vs_train_step
    )


def pretrained_agent_vs_training_cost(map_number, discrete=True, reward_type=None, num_intervals=10, alpha=0.2):
    # this doesn't actually get used, it just defines the state and action space for the agent
    train_env, map_name = read_map(number=map_number, discrete=discrete, random_start=False, reward_type=reward_type)
    size = read_grid_size(number=map_number)[0]

    agent_names = get_all_model_names(map_num=map_number)
    experiment_name = f'interval_{map_name}{map_number}' if discrete else f'continuous-interval_{map_name}{map_number}'
    subagents = [
        SacFactory.create(state_space=train_env.observation_space,
                          action_space=train_env.action_space,
                          subagent_name=name,
                          experiment_name=experiment_name,
                          discrete=discrete,
                          alpha=alpha,
                          learning_decay=0.99,
                          discount_rate=0.975,
                          max_ep_len=(size ** 2),
                          steps_per_epoch=(size ** 2) * 2,
                          start_steps=(size ** 2) * 2 * 2,
                          pi_lr=3e-4,
                          critic_lr=3e-4,
                          num_test_eps=1)
        for name in agent_names
    ]

    path_costs_vs_train_step = defaultdict(float)
    accumulated_deceptiveness_vs_train_step = defaultdict(float)
    training_interval_steps = 120000 if len(subagents) == 3 else 150000

    for i in range(num_intervals):
        # do a round of training
        for subagent in subagents:
            run_subagent(map_number, subagent, int(training_interval_steps / len(subagents)), discrete=discrete,
                         reward_type=reward_type)

        # test the agent
        rg_probs, path_cost = test_interval(map_number, agent_type=AmbiguityTypes.INTERVAL_SAC,
                                            model_config='interval_sac', discrete=discrete, reward_type=reward_type)

        # measure real goal probs at different stages to avoid high deceptiveness from agents that take lots of steps
        rg_probs = convert_to_time_density(rg_probs).values()
        accumulated_deceptiveness = sum(map(lambda p: 1 - p, rg_probs))

        print(f"accumulated deceptiveness = {accumulated_deceptiveness}")
        print(f"path cost = {path_cost}")

        path_costs_vs_train_step[(training_interval_steps / 10) * (i + 1)] = path_cost
        accumulated_deceptiveness_vs_train_step[(training_interval_steps / 10) * (i + 1)] = accumulated_deceptiveness

    append_results_to_json(
        fp=get_interval_accumulated_deceptiveness_json_path(
            agent_type=AmbiguityTypes.INTERVAL_SAC,
            discrete=discrete),
        key=f'{map_name}{map_number}',
        results=accumulated_deceptiveness_vs_train_step
    )
    append_results_to_json(
        fp=get_interval_path_cost_json_path(AmbiguityTypes.INTERVAL_SAC, discrete=discrete),
        key=f'{map_name}{map_number}',
        results=path_costs_vs_train_step
    )


def get_value_iteration_path(map_name, map_number, goal_name):
    """
    Used to instantiate the intention recognition agent
    """
    return f'/Users/alanlewis/PycharmProjects/DeceptiveReinforcementLearning/model_storage/value_iteration/{map_name}{map_number}-{goal_name}.npy'


def test_interval(map_number, agent_type='Pretrained', model_config='interval_sac', discrete=True, reward_type=None):
    env, map_name = read_map(number=map_number, random_start=False, terminate_at_any_goal=False, goal_name='rg',
                             discrete=discrete, destination_tolerance_range=2, reward_type=reward_type)
    start_state = read_start(number=map_number)
    goals, _ = read_goals(number=map_number, include_start=False, discrete=discrete)
    agent_names = get_all_model_names(map_num=map_number)

    intention_recognition = IntentionRecognitionFactory.create(discrete=discrete,
                                                               state_space=env.observation_space,
                                                               action_space=env.action_space,
                                                               # all_models=[get_value_iteration_path(map_name=map_name,
                                                               #                                      map_number=map_number,
                                                               #                                      goal_name=name)
                                                               #             for name in agent_names],
                                                               all_model_names=agent_names,
                                                               start_state=start_state,
                                                               goals=goals,
                                                               map_num=map_number)

    agent = AmbiguityFactory.create(state_space=env.observation_space,
                                    action_space=env.action_space,
                                    agent_type=agent_type,
                                    env_number=map_number,
                                    discrete=discrete,
                                    pruning_decay=0.0,
                                    pruning_constant=0.0)

    rg_probs, path_cost = agent.single_environment_run(env_detail=(env, f'{map_name}-{map_number}'),
                                                       agent_type=model_config,
                                                       discrete=discrete,
                                                       intention_recognition=intention_recognition)

    return rg_probs, path_cost


def run_subagent(num_env, agent: SacBaseAgent = None, interval_steps=30000, discrete=True, reward_type='value_table'):
    train_env, map_name = read_map(num_env, random_start=False, terminate_at_any_goal=False, goal_name=agent.name,
                                   discrete=discrete, reward_type=reward_type)
    test_env, map_name = read_map(num_env, random_start=False, terminate_at_any_goal=False, goal_name=agent.name,
                                  discrete=discrete, reward_type=reward_type)
    agent.interval_train(train_env, test_env=test_env, num_steps=interval_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparam', type=str, default='pruning_constant')
    parser.add_argument('--map_num', type=int, default=1)
    parser.add_argument('--policy', type=str, default='softmax')
    parser.add_argument('--discrete', type=bool, default=False)
    parser.add_argument('--agent_type', type=str, default='hyperparam')
    parser.add_argument('--reward_type', type=str, default='path_cost')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    map_num = args.map_num
    discrete = args.discrete
    policy = args.policy
    hyperparam = args.hyperparam
    agent_type = args.agent_type
    reward_type = args.reward_type
    alpha = args.alpha
    seed = args.seed

    print(f"map num = {map_num}")
    print(f"discrete = {discrete}")
    print(f"policy = {policy}")
    print(f"hyperparam = {hyperparam}")
    print(f"agent_type = {agent_type}")
    print(f"reward_type = {reward_type}")
    print(f"alpha = {alpha}")
    print(f"seed = {seed}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    if agent_type == 'interval_sac':
        pretrained_agent_vs_training_cost(map_number=map_num, discrete=discrete, reward_type=reward_type,
                                          num_intervals=20, alpha=alpha)
    elif agent_type == 'interval_online_sac':
        online_agent_vs_training_cost(map_number=map_num, pruning_decay=0, discrete=discrete, reward_type=reward_type,
                                      num_intervals=20, alpha=alpha)
    else:
        if policy == 'softmax':
            if hyperparam == 'pruning_constant':
                train_online_ambiguity_vs_pruning_constant(map_num=map_num, discrete=discrete, reward_type=reward_type,
                                                           alpha=alpha)
            elif hyperparam == 'pruning_decay':
                train_online_ambiguity_vs_pruning_decay(map_num=map_num, discrete=discrete, reward_type=reward_type,
                                                        alpha=alpha)
            elif hyperparam == 'tau_constant':
                train_online_ambiguity_vs_tau_constant(map_num=map_num, discrete=discrete, reward_type=reward_type,
                                                       alpha=alpha)
            elif hyperparam == 'tau_decay':
                train_online_ambiguity_vs_tau_decay(map_num=map_num, discrete=discrete, reward_type=reward_type,
                                                    alpha=alpha)
            else:
                raise ValueError("Invalid hyperparam type")
        elif policy == 'hardmax':
            run_online_ac_ambiguity(num_env=map_num,
                                    policy_type=policy,
                                    discrete=discrete)
        else:
            raise ValueError("Invalid policy type")
