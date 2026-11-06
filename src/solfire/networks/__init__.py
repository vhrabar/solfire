from .padding import pad_to_multiple
from .pretrained import DECODERS, Pretrained

# name -> model class; checkpoints record the name. The experiments register their own
# architectures here (solfire.experiments.register) so their checkpoints load as well.
ARCHITECTURES: dict[str, type] = {
    "pretrained": Pretrained,
}


def register_architecture(name: str, model_class: type) -> None:
    """
    Make an architecture available to ``train --arch`` and to :func:`~solfire.inference.load_model`.
    :param name: Name the checkpoints record
    :param model_class: ``nn.Module`` subclass built from the checkpoint's ``model_config``
    :return: None
    """
    ARCHITECTURES[name] = model_class


__all__ = [
    "ARCHITECTURES",
    "DECODERS",
    "Pretrained",
    "pad_to_multiple",
    "register_architecture",
]
