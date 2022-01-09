import multiprocessing as mp
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
from gym_minigrid.envs.deceptive import DeceptiveEnv
from python.minigrid_env_utils import SimpleObsWrapper
from python.runners.env_reader import read_map, read_grid_size, read_goals, read_start, read_name
from spinningup.spinup.algos.pytorch.sac.sac_agent import DiscreteSacAgent, ContinuousSacAgent, SacFactory
from python.path_manager import get_all_model_names
import python.path_manager as path_manager
from python.intention_recognition import IntentionRecognitionFactory
from python.data_parsing import append_results_to_json, convert_to_time_density
from display_utils import VideoViewer

sacPathManager = path_manager.Sac()
valueIterationPathManager = path_manager.ValueIteration()


def run_simple(agent_key='rg'):
    map = SimpleObsWrapper(DeceptiveEnv.load_from_file(
        fp=f'/Users/alanlewis/PycharmProjects/DeceptiveReinforcementLearning/maps/drl/empty.map',
        optcost=1,
        start_pos=(47, 47),
        real_goal=(1, 1, 'rg'),
        fake_goals=[(47, 1, 'fg1')],
        random_start=False,
        terminate_at_any_goal=False,
        goal_name=agent_key))

    train_env = deepcopy(map)
    test_env = deepcopy(map)
    train_env.seed(42)
    test_env.seed(42)

    agent = DiscreteSacAgent(train_env.observation_space,
                             train_env.action_space,
                             agent_name=agent_key,
                             experiment_name=f'ignore_from_file_simple',
                             start_steps=40000,
                             max_ep_len=49 ** 2,
                             steps_per_epoch=16000,
                             num_epochs=100,
                             policy_update_delay=1,
                             seed=42,
                             alpha=0.2,
                             polyak=0.995,
                             hidden_dimension=64,
                             critic_lr=1e-3,
                             pi_lr=1e-3)
    agent.train(train_env, test_env=test_env)


def train_subagent(map_num, agent_name, discrete=True, render=False, reward_type='value_table', dilate=False,
                   max_speed=1):
    size = read_grid_size(number=map_num)[0]
    train_env, map_name = read_map(map_num, random_start=False, terminate_at_any_goal=False, goal_name=agent_name,
                                   discrete=discrete, max_episode_steps=(size ** 2), dilate=dilate, max_speed=max_speed,
                                   reward_type=reward_type)
    test_env, map_name = read_map(map_num, random_start=False, terminate_at_any_goal=False, goal_name=agent_name,
                                  discrete=discrete, max_episode_steps=(size ** 2), dilate=dilate, max_speed=max_speed,
                                  reward_type=reward_type)
    experiment_name = f'pretrained-sac-{map_name}{map_num}' if discrete else f'continuous-pretrained-sac-{map_name}{map_num}'
    agent = SacFactory.create(state_space=train_env.observation_space,
                              action_space=train_env.action_space,
                              subagent_name=agent_name,
                              experiment_name=experiment_name,
                              discrete=discrete,
                              alpha=0.01,
                              learning_decay=0.99,
                              discount_rate=0.975,
                              max_ep_len=(size ** 2),
                              steps_per_epoch=(size ** 2) * 2,
                              start_steps=120000,
                              pi_lr=3e-4,
                              critic_lr=3e-4,
                              batch_size=100,
                              hidden_dim=64,
                              num_test_eps=1,
                              num_epochs=100)
    agent.train(train_env=train_env, test_env=test_env)


def train_subagents_parallel():
    for map_number in range(1, 2):
        discrete = False
        agent_names = get_all_model_names(map_number)
        pool = mp.Pool(len(agent_names))
        pool.starmap(train_subagent, [(map_number, name, discrete) for name in agent_names])


def run_honest_agent(map_num, discrete=True, render=False):
    env, map_name = read_map(map_num, random_start=False, terminate_at_any_goal=False, goal_name='rg',
                             discrete=discrete)
    video_viewer = VideoViewer()
    env = video_viewer.wrap_env(env=env, agent_name='rg',
                                folder=f'{path_manager.Saving.VIDEO_ROOT}/AmbiguityAgent/{map_name}{map_num}')
    agent = torch.load(
        sacPathManager.get_path(agent_type='sac', map_name=f'{map_name}{map_num}', agent_name='rg', discrete=discrete)
    )
    start_state = read_start(map_num)
    goals, _ = read_goals(map_num)
    agent_names = path_manager.get_all_model_names(map_num, discrete)

    # intention_recognition = IntentionRecognitionFactory.create(discrete=discrete,
    #                                                            state_space=env.observation_space,
    #                                                            action_space=env.action_space,
    #                                                            all_models=[valueIterationPathManager.get_path(
    #                                                                map_name=f'{map_name}{map_num}',
    #                                                                agent_name=name) for name in agent_names],
    #                                                            all_model_names=agent_names,
    #                                                            start_state=start_state,
    #                                                            goals=goals,
    #                                                            map_num=map_num)

    # debug: env.state, env.value_tables['rg'][int(env.state[1])][int(env.state[0])], env.value_tables['rg'][37][42]

    intention_recognition = None

    state_visitation_dict = defaultdict(int)
    state = env.reset()
    done = False
    path_cost = 0
    num_steps = 0
    max_steps = 2000

    while not done and num_steps < max_steps:
        if render:
            env.render()
        state_visitation_dict[str(state)] += 1
        state = torch.as_tensor(state, dtype=torch.float32)
        action = agent.act(state, deterministic=True)
        if discrete:
            action = action[0][0]
        path_cost += 1

        if intention_recognition is not None:
            _ = intention_recognition.predict_goal_probabilities(state, action)
        print('action:', action)
        next_state, reward, done, info = env.step(action)
        state = next_state
        num_steps += 1
    env.close()
    state_visitation_dict[str(state)] += 1
    if intention_recognition is not None:
        rg_probs = intention_recognition.candidate_probabilities_dict['rg']
        rg_probs_fp = path_manager.ResultPaths.get_score_json_path(agent_type='honest', discrete=discrete)
        path_cost_fp = path_manager.ResultPaths.get_path_cost_json_path(agent_type='honest', discrete=discrete)
        rg_probs_vs_time_density = convert_to_time_density(rg_probs)
        append_results_to_json(rg_probs_fp, key=f'{map_name}{map_num}', results=rg_probs_vs_time_density)
        append_results_to_json(path_cost_fp, key=f'{map_name}{map_num}', results=float(path_cost))


if __name__ == '__main__':
    # run_subagents_parallel()
    # for i in range(21, 24):
    reward_type = 'value_table'
    print(reward_type)
    train_subagent(map_num=31, agent_name='rg', discrete=True, render=False, reward_type=reward_type)
    # train_subagent(map_num=33, agent_name='rg', discrete=True, reward_type='value_table')
    # run_honest_agent(map_num=1, discrete=True, render=False)
