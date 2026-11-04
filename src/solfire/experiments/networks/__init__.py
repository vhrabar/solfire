"""
Architectures trained from scratch before the pretrained encoders took over: a plain U-Net,
a SegFormer and a TransUNet (a transformer bottleneck inside the U-Net). None beat the
pretrained U-Nets; see ``solfire/experiments/README.md``.
"""

from .Transformer import SegFormer, TransformerBottleneck
from .transunet import TransUNet
from .UNet import UNet, UNetDecoder, UNetEncoder

ARCHITECTURES = {
    "unet": UNet,
    "segformer": SegFormer,
    "transunet": TransUNet,
}

__all__ = ["ARCHITECTURES", "SegFormer", "TransUNet", "TransformerBottleneck", "UNet", "UNetDecoder", "UNetEncoder"]
