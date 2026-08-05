from .Transformer import SegFormer
from .UNet import UNet

ARCHITECTURES = {
    "unet": UNet,
    "segformer": SegFormer,
}

__all__ = ["ARCHITECTURES", "SegFormer", "UNet"]
