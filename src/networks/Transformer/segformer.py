import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoder import EncoderStage


class SegFormer(nn.Module):
    """
    Transformer encoder with an all-MLP decoder, returning per-pixel logits.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        embed_dims=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        num_heads=(1, 2, 5, 8),
        reductions=(8, 4, 2, 1),
        mlp_ratio=4.0,
        decoder_dim=256,
        dropout=0.0,
        drop_path=0.1,
    ):
        """
        :param in_channels: Channels of the input observation
        :param out_channels: Channels of the prediction, ``1`` for a binary mask
        :param embed_dims: Token width of each encoder stage
        :param depths: Number of encoder blocks in each stage
        :param num_heads: Attention heads in each stage; must divide the matching ``embed_dims`` entry
        :param reductions: Key/value downsampling factor in each stage
        :param mlp_ratio: Hidden width of every Mix-FFN as a multiple of its stage width
        :param decoder_dim: Width every stage is projected to before the decoder fuses them
        :param dropout: Dropout rate inside the encoder and before the head
        :param drop_path: Maximum stochastic depth rate, ramped linearly over the blocks
        """
        super().__init__()
        stages = len(embed_dims)
        if not all(len(setting) == stages for setting in (depths, num_heads, reductions)):
            raise ValueError("embed_dims, depths, num_heads and reductions must be the same length")

        # Stage i sees the input at 1/(4 * 2**i), and its attention pools that grid by reductions[i].
        # Anything the pair cannot divide evenly gets padded up in forward().
        self.divisor = max(4 * 2**i * reduction for i, reduction in enumerate(reductions))

        # Stochastic depth ramps from none in the first block to drop_path in the last.
        rates = torch.linspace(0, drop_path, max(1, sum(depths))).tolist()

        self.stages = nn.ModuleList()
        cursor = 0
        for stage in range(stages):
            self.stages.append(
                EncoderStage(
                    in_channels if stage == 0 else embed_dims[stage - 1],
                    embed_dims[stage],
                    num_heads[stage],
                    # The stem takes the big stride; later stages just halve.
                    patch_size=7 if stage == 0 else 3,
                    stride=4 if stage == 0 else 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    reduction=reductions[stage],
                    drop_paths=rates[cursor : cursor + depths[stage]],
                )
            )
            cursor += depths[stage]

        self.projections = nn.ModuleList(nn.Linear(dim, decoder_dim) for dim in embed_dims)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * stages, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)

        self.apply(self._init_weights)

        nn.init.normal_(self.head.weight, std=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _init_weights(module) -> None:
        """
        Initialise one submodule; transformers do not tolerate the default linear init well.
        :param module: Submodule visited by :meth:`torch.nn.Module.apply`
        :return: None
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm | nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _encode(self, x) -> list[Tensor]:
        """
        Run the staged encoder, keeping the feature map produced by every stage.
        :param x: ``(B, in_channels, H, W)`` batch of observations
        :return: One ``(B, embed_dims[i], H / 2**(i + 2), W / 2**(i + 2))`` feature map per stage
        """
        features: list[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    def _decode(self, features) -> Tensor:
        """
        Project every stage to a common width, resample to the finest grid and fuse.
        :param features: Per-stage feature maps from :meth:`_encode`
        :return: ``(B, out_channels, H / 4, W / 4)`` logits
        """
        size = features[0].shape[-2:]
        merged = []
        for feature, projection in zip(features, self.projections):
            tokens = projection(feature.flatten(2).transpose(1, 2))
            spatial = tokens.transpose(1, 2).reshape(feature.shape[0], -1, *feature.shape[-2:])
            merged.append(F.interpolate(spatial, size=size, mode="bilinear", align_corners=False))

        logits: Tensor = self.head(self.drop(self.fuse(torch.cat(merged, dim=1))))
        return logits

    def _pad(self, x: Tensor) -> Tensor:
        """
        Grow the input to the next multiple of :attr:`divisor` the stages can divide evenly.

        :param x: ``(B, in_channels, H, W)`` batch of observations
        :return: The batch, padded to a divisible size
        """
        height, width = x.shape[-2:]
        pad_height = -height % self.divisor
        pad_width = -width % self.divisor
        if not pad_height and not pad_width:
            return x
        return F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")

    def forward(self, x) -> Tensor:
        """
        Accepts any input size; sizes the stages cannot divide are padded and cropped back.

        :param x: ``(B, in_channels, H, W)`` batch of observations
        :return: ``(B, out_channels, H, W)`` logits
        """
        height, width = x.shape[-2:]
        padded = self._pad(x)

        logits = self._decode(self._encode(padded))
        logits = F.interpolate(logits, size=padded.shape[-2:], mode="bilinear", align_corners=False)
        return logits[..., :height, :width]
