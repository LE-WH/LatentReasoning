"""SFT method registry.

Follows RAGEN's environment registration pattern (ragen/env/__init__.py).
"""

from .self_training_concise import ConciseMethod
from .direct import DirectMethod

REGISTERED_METHODS = {
    "direct": DirectMethod,
    "concise": ConciseMethod,
}
