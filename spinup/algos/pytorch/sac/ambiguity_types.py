from enum import Enum


class AmbiguityTypes(str, Enum):
    SAC = 'sac'
    ONLINE_SAC = 'online_sac',
    VALUE_ITERATION = 'value_iteration',
    Q_LEARNING = 'q',
    INTERVAL_SAC = 'interval_sac',
    INTERVAL_ONLINE_SAC = 'interval_online_sac',
    PRUNING_DECAY = 'decay_param_ablation',
    PRUNING_CONSTANT = 'pruning_constant_ablation'
