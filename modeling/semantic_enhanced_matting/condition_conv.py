import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init
from typing import Any, Optional, Tuple, Type

from modeling.semantic_enhanced_matting.modeling.image_encoder import Attention
from modeling.semantic_enhanced_matting.modeling.transformer import Attention as DownAttention
from modeling.semantic_enhanced_matting.feature_fusion import PositionEmbeddingRandom as ImagePositionEmbedding
from modeling.semantic_enhanced_matting.modeling.common import MLPBlock

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    

class ConditionConv(nn.Module):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm=LayerNorm2d,
        act_layer=nn.GELU,
        conv_kernels=3,
        conv_paddings=1,
        condtition_channels = 1024
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__()

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = norm(bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            conv_kernels,
            padding=conv_paddings,
            bias=False,
        )
        self.norm2 = norm(bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = norm(out_channels)

        self.init_weight()

        self.condition_embedding = nn.Sequential(
            act_layer(),
            nn.Linear(condtition_channels, bottleneck_channels, bias=True)
        )

    def init_weight(self):
        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    # def embed_bbox_and_instance(self, bbox, instance):
    #     assert isinstance(instance, bool)

    def forward(self, x, condition):
        # [B, 64, 64, 1024]
        out = x.permute(0, 3, 1, 2)

        out = self.act1(self.norm1(self.conv1(out)))
        out = self.conv2(out) + self.condition_embedding(condition)[:, :, None, None]
        out = self.act2(self.norm2(out))
        out = self.norm3(self.conv3(out))

        out = x + out.permute(0, 2, 3, 1)
        return out


class ConditionAdd(nn.Module):
    def __init__(
        self,
        act_layer=nn.GELU,
        condtition_channels = 1024
    ):
        super().__init__()

        self.condition_embedding = nn.Sequential(
            act_layer(),
            nn.Linear(condtition_channels, condtition_channels, bias=True)
        )

    def forward(self, x, condition):
        # [B, 64, 64, 1024]
        condition = self.condition_embedding(condition)[:, None, None, :]
        return x + condition

class ConditionEmbedding(nn.Module):
    def __init__(
        self,
        condition_num = 5,
        pos_embedding_dim = 128,
        embedding_scale = 1.0,
        embedding_max_period = 10000,
        embedding_flip_sin_to_cos = True,
        embedding_downscale_freq_shift = 1.0,
        time_embed_dim = 1024,
        split_embed = False
    ):
        super().__init__()
        self.condition_num = condition_num
        self.pos_embedding_dim = pos_embedding_dim
        self.embedding_scale = embedding_scale
        self.embedding_max_period = embedding_max_period
        self.embedding_flip_sin_to_cos = embedding_flip_sin_to_cos
        self.embedding_downscale_freq_shift = embedding_downscale_freq_shift
        self.split_embed = split_embed

        if self.split_embed:
            self.linear_1 = nn.Linear(pos_embedding_dim, time_embed_dim, True)
        else:
            self.linear_1 = nn.Linear(condition_num * pos_embedding_dim, time_embed_dim, True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, True)

    def proj_embedding(self, condition):
        sample = self.linear_1(condition)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample
    
    def position_embedding(self, condition):
        # [B, 5] --> [B, 5, 128] --> [B, 5 * 128]

        assert condition.shape[-1] == self.condition_num

        half_dim = self.pos_embedding_dim // 2
        exponent = -math.log(self.embedding_max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=condition.device
        )
        exponent = exponent / (half_dim - self.embedding_downscale_freq_shift)

        emb = torch.exp(exponent)
        emb = condition[:, :, None].float() * emb[None, None, :]  # [B, 5, 1] * [1, 1, 64] --> [B, 5, 64]

        # scale embeddings
        emb = self.embedding_scale * emb

        # concat sine and cosine embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, 5, 64] --> [B, 5, 128]

        # flip sine and cosine embeddings
        if self.embedding_flip_sin_to_cos:
            emb = torch.cat([emb[:, :, half_dim:], emb[:, :, :half_dim]], dim=-1)

        # zero pad
        # if self.pos_embedding_dim % 2 == 1:
        #     emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        if self.split_embed:
            emb = emb.reshape(-1, emb.shape[-1])
        else:
            emb = emb.reshape(emb.shape[0], -1)

        return emb

    def forward(self, condition):
        condition = self.position_embedding(condition)
        condition = self.proj_embedding(condition)
        return condition.float()



class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        
        self.positional_encoding_gaussian_matrix = nn.Parameter(scale * torch.randn((2, num_pos_feats // 2)))
        # self.register_buffer(
        #     "positional_encoding_gaussian_matrix",
        #     scale * torch.randn((2, num_pos_feats)),
        # )
        point_embeddings = [nn.Embedding(1, num_pos_feats) for i in range(2)]
        self.point_embeddings = nn.ModuleList(point_embeddings)

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        coords =  self._pe_encoding(coords.to(torch.float))  # B x N x C

        coords[:, 0, :] += self.point_embeddings[0].weight
        coords[:, 1, :] += self.point_embeddings[1].weight

        return coords


class CrossSelfAttn(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, embedding_dim=1024, num_heads=4, downsample_rate=4) -> None:
        super().__init__()

        self.cross_attn = DownAttention(embedding_dim=embedding_dim, num_heads=num_heads, downsample_rate=downsample_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim=512)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.self_attn = DownAttention(embedding_dim=embedding_dim, num_heads=num_heads, downsample_rate=downsample_rate)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(self, block_feat, bbox_token, feat_pe, bbox_pe):
        B, H, W, C = block_feat.shape
        block_feat = block_feat.reshape(B, H * W, C)

        block_feat = block_feat + self.cross_attn(q=block_feat + feat_pe, k=bbox_token + bbox_pe, v=bbox_token)
        block_feat = self.norm1(block_feat)

        block_feat = block_feat + self.mlp(block_feat)
        block_feat = self.norm2(block_feat)

        concat_token = torch.concat((block_feat + feat_pe, bbox_token + bbox_pe), dim=1) 
        block_feat = block_feat + self.self_attn(q=concat_token, k=concat_token, v=concat_token)[:, :-bbox_token.shape[1]]
        block_feat = self.norm3(block_feat)
        output = block_feat.reshape(B, H, W, C)

        return output


class BBoxEmbedInteract(nn.Module):
    def __init__(
        self,
        embed_type = 'fourier',
        interact_type = 'attn',
        layer_num = 3
    ):
        super().__init__()
        assert embed_type in {'fourier', 'position', 'conv'}
        assert interact_type in {'add', 'attn', 'cross-self-attn'}
        self.embed_type = embed_type
        self.interact_type = interact_type
        self.layer_num = layer_num

        if self.embed_type == 'fourier' and self.interact_type == 'add':
            self.embed_layer = ConditionEmbedding(condition_num = 4, pos_embedding_dim = 256)
        elif self.embed_type == 'fourier':
            self.embed_layer = ConditionEmbedding(condition_num = 4, pos_embedding_dim = 256, split_embed = True)
        elif self.embed_type == 'conv':
            mask_in_chans = 16
            activation = nn.GELU
            self.embed_layer = nn.Sequential(
                nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans // 4),
                activation(),
                nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans),
                activation(),
                nn.Conv2d(mask_in_chans, 1024, kernel_size=1),
            )
        else:
            if self.interact_type == 'add':
                self.embed_layer = PositionEmbeddingRandom(num_pos_feats = 512)
            else:
                self.embed_layer = PositionEmbeddingRandom(num_pos_feats = 1024)

        self.interact_layer = nn.ModuleList()
        for _ in range(self.layer_num):
            if self.interact_type == 'attn':
                self.interact_layer.append(Attention(dim = 1024))
            elif self.interact_type == 'add' and self.embed_type != 'conv':
                self.interact_layer.append(nn.Sequential(
                    nn.GELU(),
                    nn.Linear(1024, 1024, bias=True)
                ))
            elif self.interact_type == 'cross-self-attn':
                self.interact_layer.append(CrossSelfAttn(embedding_dim=1024, num_heads=4, downsample_rate=4))

            self.position_layer = ImagePositionEmbedding(num_pos_feats=1024 // 2)

    def forward(self, block_feat, bbox, layer_index):
        # input: [B, 1, 4], [B, 64, 64, 1024]
        if layer_index == self.layer_num:
            return block_feat
        interact_layer = self.interact_layer[layer_index]

        bbox = bbox + 0.5  # Shift to center of pixel
        if self.embed_type == 'fourier' and self.interact_type == 'add':
            embedding = self.embed_layer(bbox[:, 0])  # [B, 1, 4] --> reshape [B, 4] --> [B, 1024 * 1] --> reshape [B, 1, 1024]
            embedding = embedding.reshape(embedding.shape[0], 1, -1)
        elif self.embed_type == 'fourier':
            embedding = self.embed_layer(bbox[:, 0])  # [B, 1, 4] --> reshape [B, 4] --> [B, 1024 * 4] --> reshape [B, 4, 1024]
            embedding = embedding.reshape(-1, 4, embedding.shape[-1])
        elif self.embed_type == 'conv':
            # concat mask and img as condition
            bbox_mask = torch.zeros(size=(block_feat.shape[0], 1, 256, 256), device=block_feat.device, dtype=block_feat.dtype)  # [B, 1, 512, 512]
            for i in range(bbox.shape[0]):
                l, u, r, d = bbox[i, 0, :] / 4
                bbox_mask[i, :, int(u + 0.5): int(d + 0.5), int(l + 0.5): int(r + 0.5)] = 1.0  # int(x + 0.5) = round(x)
            embedding = self.embed_layer(bbox_mask)  # [B, 1024, 64, 64]
        elif self.embed_type == 'position':
            embedding = self.embed_layer(bbox.reshape(-1, 2, 2), (1024, 1024))  # [B, 1, 4] --> reshape [B, 2, 2] --> [B, 2, 1024/512]
            if self.interact_type == 'add':
                embedding = embedding.reshape(embedding.shape[0], 1, -1)

        # add position embedding to block_feat
        pe = self.position_layer(size=(64, 64)).reshape(1, 64, 64, 1024)
        block_feat = block_feat + pe

        if self.interact_type == 'attn':
            add_token_num = embedding.shape[1]
            B, H, W, C = block_feat.shape
            block_feat = block_feat.reshape(B, H * W, C)
            concat_token = torch.concat((block_feat, embedding), dim=1)  # [B, 64 * 64 + 2, 1024]
            output_token = interact_layer.forward_token(concat_token)[:, :-add_token_num]
            output = output_token.reshape(B, H, W, C)
        elif self.embed_type == 'conv':
            output = block_feat + embedding.permute(0, 2, 3, 1)
        elif self.interact_type == 'add':
            output = interact_layer(embedding[:, None]) + block_feat
        elif self.interact_type == 'cross-self-attn':
            output = interact_layer(block_feat, embedding)

        return output
        

# reuse the position_point_embedding in prompt_encoder
class BBoxInteract(nn.Module):
    def __init__(
        self,
        position_point_embedding,
        point_weight,
        layer_num = 3,
    ):
        super().__init__()

        self.position_point_embedding = position_point_embedding
        self.point_weight = point_weight
        for _, p in self.named_parameters():
            p.requires_grad = False

        self.layer_num = layer_num
        self.input_image_size = (1024, 1024)

        self.interact_layer = nn.ModuleList()
        for _ in range(self.layer_num):
            self.interact_layer.append(CrossSelfAttn(embedding_dim=1024, num_heads=4, downsample_rate=4))
    
    @torch.no_grad()
    def get_bbox_token(self, boxes):
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.position_point_embedding.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_weight[2].weight
        corner_embedding[:, 1, :] += self.point_weight[3].weight
        corner_embedding = F.interpolate(corner_embedding[..., None], size=(1024, 1), mode='bilinear', align_corners=False)[..., 0]
        return corner_embedding  # [B, 2, 1024]
    
    @torch.no_grad()
    def get_position_embedding(self, size=(64, 64)):
        pe = self.position_point_embedding(size=size)
        pe = F.interpolate(pe.permute(1, 2, 0)[..., None], size=(1024, 1), mode='bilinear', align_corners=False)[..., 0][None]
        pe = pe.reshape(1, -1, 1024)
        return pe  # [1, 64 * 64, 1024]

    def forward(self, block_feat, bbox, layer_index):
        # input: [B, 1, 4], [B, 64, 64, 1024]
        if layer_index == self.layer_num:
            return block_feat
        interact_layer = self.interact_layer[layer_index]

        pe = self.get_position_embedding()
        bbox_token = self.get_bbox_token(bbox)

        output = interact_layer(block_feat, bbox_token, feat_pe=pe, bbox_pe=bbox_token)

        return output
        
class InOutBBoxCrossSelfAttn(nn.Module):

    def __init__(self, embedding_dim=1024, num_heads=4, downsample_rate=4) -> None:
        super().__init__()

        self.self_attn = DownAttention(embedding_dim=embedding_dim, num_heads=num_heads, downsample_rate=downsample_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim=embedding_dim // 2)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.cross_attn = DownAttention(embedding_dim=embedding_dim, num_heads=num_heads, downsample_rate=downsample_rate)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(self, in_box_token, out_box_token):

        # self-attn
        short_cut = in_box_token
        in_box_token = self.norm1(in_box_token)
        in_box_token = self.self_attn(q=in_box_token, k=in_box_token, v=in_box_token)
        in_box_token = short_cut + in_box_token

        # mlp
        in_box_token = in_box_token + self.mlp(self.norm2(in_box_token))

        # cross-attn
        short_cut = in_box_token
        in_box_token = self.norm3(in_box_token)
        in_box_token = self.cross_attn(q=in_box_token, k=out_box_token, v=out_box_token)
        in_box_token = short_cut + in_box_token

        return in_box_token


class BBoxInteractInOut(nn.Module):
    def __init__(
        self,
        num_heads = 4, 
        downsample_rate = 4,
        layer_num = 3,
    ):
        super().__init__()

        self.layer_num = layer_num
        self.input_image_size = (1024, 1024)

        self.interact_layer = nn.ModuleList()
        for _ in range(self.layer_num):
            self.interact_layer.append(InOutBBoxCrossSelfAttn(embedding_dim=1024, num_heads=num_heads, downsample_rate=downsample_rate))

    def forward(self, block_feat, bbox, layer_index):

        # input: [B, 1, 4], [B, 64, 64, 1024]
        if layer_index == self.layer_num:
            return block_feat
        interact_layer = self.interact_layer[layer_index]

        # split_in_out_bbox_token
        bbox = torch.round(bbox / self.input_image_size[0] * (block_feat.shape[1] - 1)).int()
        for i in range(block_feat.shape[0]):
            in_bbox_mask = torch.zeros((block_feat.shape[1], block_feat.shape[2]), dtype=bool, device=bbox.device)
            in_bbox_mask[bbox[i, 0, 1]: bbox[i, 0, 3], bbox[i, 0, 0]: bbox[i, 0, 2]] = True
            in_bbox_token = block_feat[i: i + 1, in_bbox_mask, :]
            out_bbox_token = block_feat[i: i + 1, ~in_bbox_mask, :]
            block_feat[i, in_bbox_mask, :] = interact_layer(in_bbox_token, out_bbox_token)

        return block_feat


if __name__ == '__main__':
    # emded = ConditionEmbedding()
    # input = torch.tensor([[100, 200, 300, 400, 512], [100, 200, 300, 400, 1024]])
    # print(input.shape)
    # output = emded(input)  # [B, 5] --> [B, 5 * 128] --> [B, 1024]

    emded = BBoxEmbedInteract(
        embed_type = 'position',
        interact_type = 'cross-self-attn'
    )
    input = torch.tensor([[[100, 200, 300, 400]], [[100, 200, 300, 400]]])  # [B, 1, 4]
    print(input.shape)
    output = emded(torch.randn((2, 64, 64, 1024)), input)  # [B, 5] --> [B, 5 * 128] --> [B, 1024]