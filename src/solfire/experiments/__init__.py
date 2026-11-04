"""
Experiments that were tried on top of the final method and dropped, kept runnable.

Nothing in the core imports this package. Importing it registers the experiments'
architectures (``unet``, ``segformer``, ``transunet``) and preprocessing (``clahe``) with the
core, so their checkpoints train and load; ``solfire-lab`` (:mod:`.main`) adds their options
and commands. What each one was and how it did: ``README.md`` next to this file.
"""

from solfire.data.preprocess import register_preprocessing
from solfire.networks import register_architecture

from .data.clahe import clahe
from .networks import ARCHITECTURES

for _name, _model in ARCHITECTURES.items():
    register_architecture(_name, _model)
register_preprocessing("clahe", clahe)
