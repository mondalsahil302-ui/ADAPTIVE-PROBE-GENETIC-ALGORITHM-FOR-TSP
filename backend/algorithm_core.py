"""
Backward compatibility bridge forwarding to backend.optimizer and backend.config.
"""
from backend.config import GAConfig, make_config
from backend.optimizer.models import *
from backend.optimizer.heuristics import *
from backend.optimizer.lkh import *
from backend.optimizer.lkh import _solve_elkai_internal, _solve_lkh_external
from backend.optimizer.core import *

