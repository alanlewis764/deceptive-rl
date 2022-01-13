import argparse

import numpy as np
import torch

from ambiguity.ambiguity import AmbiguityFactory
from gym_minigrid.env_reader import read_map, read_start, read_goals, get_all_model_names
from intention_recognition.intention_recognition import IntentionRecognitionFactory
from gym_minigrid.monitor import Monitor
from ambiguity.ambiguity_types import AmbiguityTypes


def wrap_env(env, file_name, folder, force=True):
    print(f'STORING IN FOLDER: {folder}/{file_name}')
    env = Monitor(env, F'{folder}/{file_name}', force=force)
    return env


def run_ambiguity(env_number, agent_type='sac', discrete=True, decay_param=0.0, pruning_constant=-100., tau_decay=0.975,
                  tau_constant=1., measure='real_goal_probs', render=False, folder='video', file_name='online'):
    env, map_name = read_map(number=env_number, discrete=discrete)
    env = wrap_env(env=env, file_name=file_name, folder=folder)
    start_state = read_start(env_number)
    goals, _ = read_goals(env_number)

    agent_names = get_all_model_names(env_number)

    intention_recognition = IntentionRecognitionFactory.create(discrete=discrete,
                                                               state_space=env.observation_space,
                                                               action_space=env.action_space,
                                                               all_model_names=agent_names,
                                                               start_state=start_state,
                                                               goals=goals,
                                                               map_num=env_number)
    # correct agent given agent_type
    agent = AmbiguityFactory.create(state_space=env.observation_space,
                                    action_space=env.action_space,
                                    agent_type=agent_type,
                                    env_number=env_number,
                                    discrete=discrete,
                                    pruning_decay=decay_param,
                                    pruning_constant=pruning_constant,
                                    tau_decay=tau_decay,
                                    tau_constant=tau_constant)

    agent.single_environment_run(env_detail=(env, f'{map_name}-{env_number}'),
                                 agent_type=agent_type,
                                 discrete=discrete,
                                 decay_param=decay_param,
                                 pruning_constant=pruning_constant,
                                 tau_decay=tau_decay,
                                 tau_constant=tau_constant,
                                 intention_recognition=intention_recognition,
                                 measure=measure,
                                 render=render)
    env.close()


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
        run_ambiguity(env_number=map_num,
                      agent_type=AmbiguityTypes.ONLINE_SAC,
                      discrete=discrete,
                      pruning_constant=0,
                      file_name=agent)
    elif agent == 'pre-trained':
        run_ambiguity(env_number=map_num,
                      agent_type=AmbiguityTypes.SAC,
                      discrete=discrete,
                      pruning_constant=0,
                      file_name=agent)
    else:
        raise ValueError("Not a valid ambiguity agent")
