from .padding import pad_to_multiple
from .Transformer import SegFormer, TransformerBottleneck
from .transunet import TransUNet
from .UNet import UNet, UNetDecoder, UNetEncoder

ARCHITECTURES = {
    "unet": UNet,
    "segformer": SegFormer,
    "transunet": TransUNet,
}

__all__ = [
    "ARCHITECTURES",
    "SegFormer",
    "TransUNet",
    "TransformerBottleneck",
    "UNet",
    "UNetDecoder",
    "UNetEncoder",
    "pad_to_multiple",
]
