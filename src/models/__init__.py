# Models package
from enum import Enum

class ModelRole(Enum):
    """
    Role of a model in the system.
    Determines which RunModes it is allowed to operate in.
    """
    TRAINING_ONLY = "TRAINING_ONLY"  # Only for training/labeling
    FROZEN_EVAL = "FROZEN_EVAL"      # Validated model for metrics
    REPLAY_ONLY = "REPLAY_ONLY"      # Specifically for simulation
    SCAN_ASSIST = "SCAN_ASSIST"      # Low-confidence signal generator

"""Neural network architectures and training."""
