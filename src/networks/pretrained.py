import segmentation_models_pytorch as smp
from torch import Tensor, nn

from .padding import pad_to_multiple

DECODERS = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "fpn": smp.FPN,
    "upernet": smp.UPerNet,
}

# ImageNet statistics averaged over the colour channels
IMAGENET_MEAN = 0.449
IMAGENET_STD = 0.226


class Pretrained(nn.Module):
    """
    A segmentation_models_pytorch decoder on top of an ImageNet encoder.

    The encoder's first convolution is collapsed to one channel by smp itself, which
    sums the RGB kernels, so the grayscale observation only has to be standardised the
    way the encoder saw its images in pretraining.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        decoder="unet",
        encoder="resnet34",
        encoder_weights="imagenet",
        encoder_depth=5,
    ):
        """
        :param in_channels: Channels of the input observation
        :param out_channels: Channels of the prediction, ``1`` for a binary mask
        :param decoder: Key of :data:`DECODERS`
        :param encoder: Encoder name as smp spells it, ``tu-`` prefixed for timm models
        :param encoder_weights: Pretrained weights to start from, or ``None`` for random init;
            a checkpoint is always rebuilt with ``None`` since its state dict carries the weights
        :param encoder_depth: Number of encoder stages the decoder reads from
        """
        super().__init__()
        self.net = DECODERS[decoder](
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            in_channels=in_channels,
            classes=out_channels,
        )
        self.divisor = 2**encoder_depth

    @property
    def encoder(self) -> nn.Module:
        """
        :return: The pretrained part of the network, for a separate learning rate
        """
        encoder: nn.Module = self.net.encoder
        return encoder

    def forward(self, x) -> Tensor:
        """
        Accepts any input size; sizes the encoder cannot halve are padded and cropped back.

        :param x: ``(B, in_channels, H, W)`` batch of observations in ``[0, 1]``
        :return: ``(B, out_channels, H, W)`` logits
        """
        height, width = x.shape[-2:]
        x = pad_to_multiple((x - IMAGENET_MEAN) / IMAGENET_STD, self.divisor)
        logits: Tensor = self.net(x)
        return logits[..., :height, :width]
