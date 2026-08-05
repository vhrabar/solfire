from .Transformer import SegFormer
from .transunet import TransUNet
from .UNet import UNet

ARCHITECTURES = {
    "unet": UNet,
    "segformer": SegFormer,
    "transunet": TransUNet,
}

__all__ = ["ARCHITECTURES", "SegFormer", "TransUNet", "UNet"]
