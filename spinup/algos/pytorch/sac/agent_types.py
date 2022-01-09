from enum import Enum


class AmbiguityTypes(str, Enum):
    SAC = 'sac'
    ONLINE_SAC = 'online_sac',
    VALUE_ITERATION = 'value_iteration',
    Q_LEARNING = 'q',
    INTERVAL_SAC = 'interval_sac',
    INTERVAL_ONLINE_SAC = 'interval_online_sac',
    PRUNING_DECAY = 'decay_param_ablation',
    PRUNING_CONSTANT = 'pruning_constant_ablation',
    TAU_CONSTANT = 'tau_constant_ablation'
    TAU_DECAY = 'tau_decay_ablation'


class DgacTypes(str, Enum):
    DGAC = 'dgac',
    STATIC = 'static',
    STATIC_ENTROPY = 'static_entropy',
    STATIC_EXAGGERATION = 'static_exaggeration',
    PRETRAINED = 'pretrained'
    ONLINE = 'online',
    VALUE_ITERATION = 'value_iteration'
