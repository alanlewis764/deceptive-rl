import json
import os
import pandas as pd
import pathlib
from collections import Counter
from spinup.algos.pytorch.sac.ambiguity_types import AmbiguityTypes


def get_interval_accumulated_deceptiveness_json_path(agent_type, discrete):
    if agent_type == AmbiguityTypes.INTERVAL_SAC:
        fn = 'interval_pretrained_ambiguity_accumulated_deceptiveness'
    elif agent_type == AmbiguityTypes.INTERVAL_ONLINE_SAC:
        fn = 'interval_online_ambiguity_accumulated_deceptiveness'
    else:
        raise ValueError("No such interval agent :(")
    folder = get_folder(agent_type=agent_type, discrete=discrete)
    return folder + fn + '.json'


def get_interval_path_cost_json_path(agent_type, discrete):
    if agent_type == AmbiguityTypes.INTERVAL_SAC:
        fn = 'interval_pretrained_ambiguity_path_cost'
    elif agent_type == AmbiguityTypes.INTERVAL_ONLINE_SAC:
        fn = 'interval_online_ambiguity_path_cost'
    else:
        raise ValueError("No such interval agent :(")
    folder = get_folder(agent_type=agent_type, discrete=discrete)
    return folder + fn + '.json'


def get_folder(agent_type, discrete):
    root = '/data/projects/punim1607/spinningup/results/'
    if agent_type == AmbiguityTypes.PRUNING_DECAY:
        folder = f'{root}decay_param/'
    elif agent_type == AmbiguityTypes.PRUNING_CONSTANT:
        folder = f'{root}pruning_constant/'
    elif agent_type == AmbiguityTypes.TAU_DECAY:
        folder = f'{root}tau_decay/'
    elif agent_type == AmbiguityTypes.TAU_CONSTANT:
        folder = f'{root}tau_constant/'
    elif discrete:
        folder = f'{root}'
    else:
        folder = f'{root}continuous/'
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def map_path_to_dict(fp):
    with open(fp) as f:
        my_data = json.load(f)
        # convert the keys to state tuple
        my_data = dict(map(lambda kv: ((eval(kv[0])), kv[1]), my_data.items()))
    return my_data


def append_results_to_json(fp, key, results):
    if not os.path.exists(fp):
        data = {}
    else:
        with open(fp) as f:
            data = json.load(f)
    data.update({key: results})
    with open(fp, 'w') as f:
        json.dump(data, f)


def convert_to_time_density(arr):
    num_time_steps = len(arr)
    out_map = {}
    for i in range(10):
        density = i / 10
        out_map.update({density: arr[int(num_time_steps * density)]})
    return out_map


def merge_dicts(dicts):
    result = Counter()
    for dict in dicts:
        result += Counter(dict)
    return result


def read_json_as_dict(fp):
    with open(fp, 'r') as f:
        return json.load(fp=f)
