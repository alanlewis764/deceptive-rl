# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Algorithms
from spinup.algos.pytorch.sac.sac_agent import DiscreteSacAgent as sac_pytorch

# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__