from .padding import pad_to_multiple
from .pretrained import DECODERS, Pretrained
from .Transformer import SegFormer, TransformerBottleneck
from .transunet import TransUNet
from .UNet import UNet, UNetDecoder, UNetEncoder

ARCHITECTURES = {
    "unet": UNet,
    "segformer": SegFormer,
    "transunet": TransUNet,
    "pretrained": Pretrained,
}

__all__ = [
    "ARCHITECTURES",
    "DECODERS",
    "Pretrained",
    "SegFormer",
    "TransUNet",
    "TransformerBottleneck",
    "UNet",
    "UNetDecoder",
    "UNetEncoder",
    "pad_to_multiple",
]
