import torch
import torch.nn as nn
from typing import Type, Optional, Tuple
import numpy as np

from .modeling.transformer import Attention
from .modeling.common import MLPBlock
# from modeling.transformer import Attention
# from modeling.common import MLPBlock


class MutualCrossAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 1024,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 4,
    ) -> None:
        super().__init__()

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.norm3 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

    def forward(self, queries, keys, query_pe=None, key_pe=None):

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe if query_pe is not None else queries
        k = keys + key_pe if key_pe is not None else keys
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm2(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe if query_pe is not None else queries
        k = keys + key_pe if key_pe is not None else keys
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm3(keys)

        return queries, keys


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        # return pe.permute(2, 0, 1)  # C x H x W
        return pe.reshape(h * w, -1)[None]  # 1 x (H x W) x C
    

class FeatureFusion(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        input_compression_ratio=1,
        attn_compression_ratio=4,
        features_num=4,
        w_pe=True,
    ):
        super().__init__()

        self.input_compression_ratio = input_compression_ratio
        if self.input_compression_ratio != 1:
            self.mlp_in = nn.ModuleList([nn.Sequential(
                nn.Linear(in_channels, in_channels // input_compression_ratio),
                # activation(),
                # nn.Linear(embedding_dim // compression_ratio, embedding_dim // compression_ratio)
            ) for _ in range(features_num)])

            self.mlp_out = nn.ModuleList([nn.Sequential(
                nn.Linear(in_channels // input_compression_ratio, in_channels),
                # activation(),
                # nn.Linear(embedding_dim, embedding_dim)
            ) for _ in range(features_num)])

        in_channels = in_channels // input_compression_ratio
        self.mutual_cross_attn = nn.ModuleList([
            MutualCrossAttention(embedding_dim=in_channels, mlp_dim=in_channels // attn_compression_ratio, attention_downsample_rate=attn_compression_ratio) for _ in range(features_num - 1)
        ])
        self.w_pe = w_pe
        if self.w_pe:
            # no grad
            self.get_pe = PositionEmbeddingRandom(in_channels // 2)
            with torch.no_grad():
                self.pe = self.get_pe(size=(64, 64))

    def forward(self, features):
        # [B, 64, 64, 1024] x 4
        
        b, h, w, _ = features[0].shape
        for i in range(len(features)):
            features[i] = features[i].reshape(b, h * w, -1)
            if self.input_compression_ratio != 1:
                features[i] = self.mlp_in[i](features[i])

        for i in range(len(features) - 1):   
            features[i], features[i + 1] = self.mutual_cross_attn[i](features[i], features[i + 1], self.pe, self.pe)

        for i in range(len(features)):
            features[i] = features[i].reshape(b, h, w, -1)
            if self.input_compression_ratio != 1:
                features[i] = self.mlp_out[i](features[i])

        return features


if __name__ == '__main__':

    import typing
    from collections import defaultdict
    import tabulate
    from torch import nn


    def parameter_count(model: nn.Module, trainable_only: bool = False) -> typing.DefaultDict[str, int]:
        """
        Count parameters of a model and its submodules.

        Args:
            model: a torch module

        Returns:
            dict (str-> int): the key is either a parameter name or a module name.
            The value is the number of elements in the parameter, or in all
            parameters of the module. The key "" corresponds to the total
            number of parameters of the model.
        """
        r = defaultdict(int)
        for name, prm in model.named_parameters():
            if trainable_only:
                if not prm.requires_grad:
                    continue
            size = prm.numel()
            name = name.split(".")
            for k in range(0, len(name) + 1):
                prefix = ".".join(name[:k])
                r[prefix] += size
        return r


    def parameter_count_table(
        model: nn.Module, max_depth: int = 3, trainable_only: bool = False
    ) -> str:
        """
        Format the parameter count of the model (and its submodules or parameters)
        in a nice table. It looks like this:

        ::

            | name                            | #elements or shape   |
            |:--------------------------------|:---------------------|
            | model                           | 37.9M                |
            |  backbone                       |  31.5M               |
            |   backbone.fpn_lateral3         |   0.1M               |
            |    backbone.fpn_lateral3.weight |    (256, 512, 1, 1)  |
            |    backbone.fpn_lateral3.bias   |    (256,)            |
            |   backbone.fpn_output3          |   0.6M               |
            |    backbone.fpn_output3.weight  |    (256, 256, 3, 3)  |
            |    backbone.fpn_output3.bias    |    (256,)            |
            |   backbone.fpn_lateral4         |   0.3M               |
            |    backbone.fpn_lateral4.weight |    (256, 1024, 1, 1) |
            |    backbone.fpn_lateral4.bias   |    (256,)            |
            |   backbone.fpn_output4          |   0.6M               |
            |    backbone.fpn_output4.weight  |    (256, 256, 3, 3)  |
            |    backbone.fpn_output4.bias    |    (256,)            |
            |   backbone.fpn_lateral5         |   0.5M               |
            |    backbone.fpn_lateral5.weight |    (256, 2048, 1, 1) |
            |    backbone.fpn_lateral5.bias   |    (256,)            |
            |   backbone.fpn_output5          |   0.6M               |
            |    backbone.fpn_output5.weight  |    (256, 256, 3, 3)  |
            |    backbone.fpn_output5.bias    |    (256,)            |
            |   backbone.top_block            |   5.3M               |
            |    backbone.top_block.p6        |    4.7M              |
            |    backbone.top_block.p7        |    0.6M              |
            |   backbone.bottom_up            |   23.5M              |
            |    backbone.bottom_up.stem      |    9.4K              |
            |    backbone.bottom_up.res2      |    0.2M              |
            |    backbone.bottom_up.res3      |    1.2M              |
            |    backbone.bottom_up.res4      |    7.1M              |
            |    backbone.bottom_up.res5      |    14.9M             |
            |    ......                       |    .....             |

        Args:
            model: a torch module
            max_depth (int): maximum depth to recursively print submodules or
                parameters

        Returns:
            str: the table to be printed
        """
        count: typing.DefaultDict[str, int] = parameter_count(model, trainable_only)
        # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
        param_shape: typing.Dict[str, typing.Tuple] = {
            k: tuple(v.shape) for k, v in model.named_parameters()
        }

        # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
        table: typing.List[typing.Tuple] = []

        def format_size(x: int) -> str:
            if x > 1e8:
                return "{:.1f}G".format(x / 1e9)
            if x > 1e5:
                return "{:.1f}M".format(x / 1e6)
            if x > 1e2:
                return "{:.1f}K".format(x / 1e3)
            return str(x)

        def fill(lvl: int, prefix: str) -> None:
            if lvl >= max_depth:
                return
            for name, v in count.items():
                if name.count(".") == lvl and name.startswith(prefix):
                    indent = " " * (lvl + 1)
                    if name in param_shape:
                        table.append((indent + name, indent + str(param_shape[name])))
                    else:
                        table.append((indent + name, indent + format_size(v)))
                        fill(lvl + 1, name + ".")

        table.append(("model", format_size(count.pop(""))))
        fill(0, "")

        old_ws = tabulate.PRESERVE_WHITESPACE
        tabulate.PRESERVE_WHITESPACE = True
        tab = tabulate.tabulate(table, headers=["name", "#elements or shape"], tablefmt="pipe")
        tabulate.PRESERVE_WHITESPACE = old_ws
        return tab

    feature_fusion = FeatureFusion(in_channels=1024, attn_compression_ratio=8)
    print("All parameters: \n" + parameter_count_table(feature_fusion, max_depth=8))
    features = [torch.randn(2, 64, 64, 1024) for _ in range(4)]
    out = feature_fusion(features)
    for i in out:
        print(i.shape)
    print('done')
