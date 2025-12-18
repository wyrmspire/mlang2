"""
Core Enums
Shared enumerations used across the codebase.
These are in a separate module to avoid circular imports.
"""

from enum import Enum


class RunMode(Enum):
    """
    Execution mode for the system.
    
    Controls what operations are permitted:
    - TRAIN: Can peek at future data for labeling, can learn, cannot trade
    - REPLAY: Cannot peek future, cannot learn, can simulate trades
    - SCAN: Cannot peek future, cannot learn, cannot trade (read-only analysis)
    """
    TRAIN = "TRAIN"
    REPLAY = "REPLAY"
    SCAN = "SCAN"


class ModelRole(Enum):
    """
    Role of a model in the system.
    Determines which RunModes it is allowed to operate in.
    """
    TRAINING_ONLY = "TRAINING_ONLY"  # Only for training/labeling
    FROZEN_EVAL = "FROZEN_EVAL"      # Validated model for metrics
    REPLAY_ONLY = "REPLAY_ONLY"      # Specifically for simulation
    SCAN_ASSIST = "SCAN_ASSIST"      # Low-confidence signal generator
