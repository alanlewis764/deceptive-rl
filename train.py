import argparse
import multiprocessing as mp
import numpy as np
import torch

from ambiguity.ambiguity import OnlineACAmbiguityAgent
from gym_minigrid.env_reader import read_map, read_grid_size, get_all_model_names, get_all_models
from subagent.algos.sac.sac_agent import SacFactory


def train_online_ambiguity(map_num, discrete=True):
    size = read_grid_size(number=map_num)[0]
    the_train_env, map_name = read_map(map_num, discrete=discrete, max_episode_steps=(size ** 2))
    the_test_env, map_name = read_map(map_num, discrete=discrete, max_episode_steps=(size ** 2))

    experiment_name = f'{map_name}{map_num}-online-ac-softmax' if discrete else f'{map_name}{map_num}-online-ac-softmax-continuous'
    max_ep_length = size ** 2
    all_models = get_all_models(map_num=map_num)
    all_model_names = get_all_model_names(map_num=map_num)

    agent = OnlineACAmbiguityAgent(
        state_space=the_train_env.observation_space,
        action_space=the_train_env.action_space,
        name=f'OnlineACContinuous',
        all_models=all_models,
        all_model_names=all_model_names,
        env=the_test_env,
        experiment_name=experiment_name,
        adaptive_pruning_constant=0,
        discrete=discrete,
        max_ep_len=max_ep_length,
        steps_per_epoch=max_ep_length * 2,
        start_steps=max_ep_length * 8,
        tau=1.0,
        tau_decay=0.9,
        num_epochs=100
    )

    agent.train(the_train_env)


def train_subagent(map_num, agent_name, discrete=True):
    size = read_grid_size(number=map_num)[0]
    train_env, map_name = read_map(map_num, discrete=discrete, max_episode_steps=(size ** 2))
    test_env, map_name = read_map(map_num, discrete=discrete, max_episode_steps=(size ** 2))
    experiment_name = f'pretrained-sac-{map_name}{map_num}' if discrete else f'continuous-pretrained-sac-{map_name}{map_num}'
    agent = SacFactory.create(state_space=train_env.observation_space,
                              action_space=train_env.action_space,
                              subagent_name=agent_name,
                              experiment_name=experiment_name,
                              discrete=discrete,
                              learning_decay=0.99,
                              discount_rate=0.975,
                              max_ep_len=(size ** 2),
                              steps_per_epoch=(size ** 2) * 2,
                              start_steps=120000,
                              num_epochs=100)
    agent.train(train_env=train_env, test_env=test_env)


def train_pretrained_ambiguity(map_num, discrete=True):
    """
    Since this is the pre-trained agent, we can just train each honest agent individually. To speed thinks up we can use
    multi-processing to parallelize it.
    """
    agent_names = get_all_model_names(map_num=map_num)
    pool = mp.Pool(len(agent_names))
    pool.starmap(train_subagent, [(map_num, agent_name, discrete) for agent_name in agent_names])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='online')
    parser.add_argument('--map_num', type=int, default=1)
    parser.add_argument('--action_space', type=str, default='discrete')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    map_num = args.map_num
    action_space = args.action_space
    agent = args.agent
    seed = args.seed
    discrete = action_space == 'discrete'
    np.random.seed(seed)
    torch.manual_seed(seed)

    if agent == 'online':
        train_online_ambiguity(map_num=map_num, discrete=discrete)
    elif agent == 'pre-trained':
        train_pretrained_ambiguity(map_num=map_num, discrete=discrete)
    else:
        raise ValueError("Not a valid ambiguity agent")
