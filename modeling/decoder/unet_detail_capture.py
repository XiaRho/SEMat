import cv2
import torch
from torch import nn
from torch.nn import functional as F
# from nnMorpho.binary_operators import erosion
from detectron2.layers.batch_norm import NaiveSyncBatchNorm


class GenTrimapTorch(object):
    def __init__(self, max_kernal=200):
        self.max_kernal = max_kernal
        self.erosion_kernels = [None] + [torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))).float().cuda() for size in range(1, self.max_kernal)]

    def __call__(self, mask, kernel_size):
        
        fg_width = kernel_size
        bg_width = kernel_size

        fg_mask = mask
        bg_mask = 1 - mask

        fg_mask = erosion(fg_mask, self.erosion_kernels[fg_width], border='a')
        bg_mask = erosion(bg_mask, self.erosion_kernels[bg_width], border='a')

        trimap = torch.ones_like(mask) * 0.5
        trimap[fg_mask == 1] = 1.0
        trimap[bg_mask == 1] = 0.0

        return trimap


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


class BasicDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, res = True, norm=LayerNorm2d, block_num=1, kernel_size=3):
        super().__init__()

        self.res = res
        self.basic_layer = nn.ModuleList()
        for i in range(block_num):
            if i == 0:
                basic_layer_in_ch = in_channel
                stride = 2
            else:
                basic_layer_in_ch = out_channel
                stride = 1
                self.basic_layer.append(nn.GELU())
            self.basic_layer.append(nn.Sequential(
                nn.Conv2d(basic_layer_in_ch, out_channel, kernel_size, stride, kernel_size // 2), 
                norm(out_channel),
                nn.GELU(),
                nn.Conv2d(out_channel, out_channel, kernel_size, 1, kernel_size // 2), 
                norm(out_channel),
            ))
        self.act = nn.GELU()

        if self.res:
            self.res_layer = nn.Conv2d(in_channel, out_channel, kernel_size, 2, kernel_size // 2)

    def forward(self, x):

        if self.res:
            identity = self.res_layer(x)
        else:
            identity = F.interpolate(x, size=(out.shape[-2], out.shape[-1]), mode='bilinear', align_corners=False)

        out = x
        for layer in self.basic_layer:
            out = layer(out)
        
        out = out + identity
        out = self.act(out)

        return out


class BasicUpBlock(nn.Module):

    def __init__( self, in_channel, out_channel, res = True, skip_connect = 'concat', norm=LayerNorm2d, block_num=1, kernel_size=3):
        super().__init__()
        assert skip_connect in {'sum', 'concat'}

        self.res = res
        self.skip_connect = skip_connect
        self.basic_layer = nn.ModuleList()
        for i in range(block_num):
            if i == 0:
                basic_layer_in_ch = in_channel
                first_conv = nn.ConvTranspose2d(basic_layer_in_ch, out_channel, 2, 2)
            else:
                basic_layer_in_ch = out_channel
                first_conv = nn.Conv2d(out_channel, out_channel, kernel_size, 1, kernel_size // 2)
                self.basic_layer.append(nn.GELU())
            self.basic_layer.append(nn.Sequential(
                first_conv, 
                norm(out_channel),
                nn.GELU(),
                nn.Conv2d(out_channel, out_channel, kernel_size, 1, kernel_size // 2), 
                norm(out_channel),
            ))
        self.act = nn.GELU()

        if self.res:
            self.res_layer = nn.Conv2d(in_channel, out_channel, kernel_size, 1, kernel_size // 2)


    def forward(self, x, skip_feat, concat_feat=None):

        if self.skip_connect == 'sum':
            x = x + skip_feat
        else:
            x = torch.concat((x, skip_feat), dim=1)

        if concat_feat is not None:
            x = torch.concat((x, concat_feat), dim=1)

        out = x
        for layer in self.basic_layer:
            out = layer(out)
        # out = self.basic_layer(x)
        
        identity = F.interpolate(x, size=(out.shape[-2], out.shape[-1]), mode='bilinear', align_corners=False)
        if self.res:
            identity = self.res_layer(identity)

        out = out + identity
        out = self.act(out)

        return out
    


class DetailUNet(nn.Module):
    def __init__(
        self,
        img_feat_in = 4,
        vit_early_feat_in = 768,
        matting_feat_in = 5,
        downsample_in_out = [(4, 32), (32, 64), (64, 128), (128, 256)],
        upsample_in_out = [(256, 128), (128, 64), (64, 32), (32, 16)],
        matting_head_in = 16,
        skip_connect = 'sum',
        norm_type = 'LN',
    ):
        super().__init__()

        assert len(downsample_in_out) == len(upsample_in_out)
        downsample_in_out[0] = (img_feat_in, downsample_in_out[0][1])

        assert norm_type in {'BN', 'LN', 'SyncBN'}
        if norm_type == 'BN':
            self.norm = torch.nn.BatchNorm2d
        elif norm_type == 'SyncBN':
            self.norm = NaiveSyncBatchNorm
        else:
            self.norm = LayerNorm2d

        self.down_blks = nn.ModuleList()
        for in_ch, out_ch in downsample_in_out:
            self.down_blks.append(
                BasicDownBlock(in_ch, out_ch, norm=self.norm)
            )
        
        self.mid_layer = nn.Sequential(
            nn.Conv2d(vit_early_feat_in, downsample_in_out[-1][1], 1, 1), 
            self.norm(downsample_in_out[-1][1]),
            nn.GELU(),
        )

        self.up_blks = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(upsample_in_out):
            if i == 2:
                in_ch += matting_feat_in
            self.up_blks.append(
                BasicUpBlock(in_ch, out_ch, skip_connect=skip_connect, norm=self.norm)
            )

        self.matting_head = nn.Conv2d(matting_head_in, 1, 3, 1, 1)


    def forward(self, x, vit_early_feat, matting_feat, return_alpha_logits=False):
        details = []
        dfeatures = x

        for i in range(len(self.down_blks)):
            dfeatures = self.down_blks[i](dfeatures)
            details.append(dfeatures)

        out = self.mid_layer(vit_early_feat)
        for i in range(len(self.up_blks)):
            if i == 2:
                out = self.up_blks[i](out, details[-i - 1], matting_feat)
            else:
                out = self.up_blks[i](out, details[-i - 1])
        alpha = self.matting_head(out)
        if return_alpha_logits:
            return alpha, out
        else:
            return alpha
    

class MattingDetailDecoder(nn.Module):
    def __init__(
        self,
        img_feat_in = 4,
        vit_intern_feat_in = 1024,
        vit_intern_feat_index = [0, 1, 2, 3],
        downsample_in_out = [(4, 32), (32, 64), (64, 128), (128, 256)],
        upsample_in_out = [(256, 128), (128, 64), (64, 32), (32, 16)],
        matting_head_in = 16,
        skip_connect = 'sum',
        norm_type = 'BN',
        norm_mask_logits = 6.5,
        with_trimap = False,
        min_kernel_size = 20,
        kernel_div = 10,
        concat_gen_trimap = False,
        wo_hq_features = False,
        block_num = 1,
        wo_big_kernel = False,
        sam2_multi_scale_feates = False,
    ):
        super().__init__()

        assert len(downsample_in_out) == len(upsample_in_out)
        assert skip_connect in {'sum', 'concat'}
        downsample_in_out[0] = (img_feat_in, downsample_in_out[0][1])
        
        self.vit_intern_feat_in = vit_intern_feat_in
        self.vit_intern_feat_index = vit_intern_feat_index
        self.norm_mask_logits = norm_mask_logits
        self.with_trimap = with_trimap
        self.min_kernel_size = min_kernel_size
        self.kernel_div = kernel_div
        self.concat_gen_trimap = concat_gen_trimap
        self.wo_hq_features = wo_hq_features
        self.block_num = block_num
        self.wo_big_kernel = wo_big_kernel
        self.sam2_multi_scale_feates = sam2_multi_scale_feates
        if self.sam2_multi_scale_feates:
            assert downsample_in_out[0][0] == 6
            downsample_in_out = [(4, 32), (32, 64), (64 + 32, 128), (128 + 64, 256)]
            upsample_in_out = [(256, 128), (128, 64), (64, 32), (32, 16)]

        if self.with_trimap and not self.concat_gen_trimap:
            self.gen_trimap = GenTrimapTorch()
        assert norm_type in {'BN', 'LN', 'SyncBN'}
        if norm_type == 'BN':
            self.norm = torch.nn.BatchNorm2d
        elif norm_type == 'SyncBN':
            self.norm = NaiveSyncBatchNorm
        else:
            self.norm = LayerNorm2d

        if self.block_num >= 2 and not self.wo_big_kernel:
            self.big_kernel_process = nn.Sequential(
                nn.Conv2d(img_feat_in, 16, kernel_size=13, stride=1, padding=6), 
                self.norm(16),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=13, stride=1, padding=6), 
                self.norm(32),
                nn.GELU(),
            )
            downsample_in_out[0] = (32, downsample_in_out[0][1])

        if not self.sam2_multi_scale_feates:
            self.vit_feat_proj = nn.ModuleDict()
            for idx in self.vit_intern_feat_index:
                self.vit_feat_proj[str(idx)] = nn.Conv2d(self.vit_intern_feat_in, self.vit_intern_feat_in // len(self.vit_intern_feat_index), 1, 1)
        self.vit_feat_aggregation = nn.Sequential(
            nn.Conv2d(self.vit_intern_feat_in // len(self.vit_intern_feat_index) * len(self.vit_intern_feat_index), downsample_in_out[-1][1], 3, 1, 1), 
            self.norm(downsample_in_out[-1][1]),
            nn.GELU(),
        )

        self.down_blks = nn.ModuleList()
        for in_ch, out_ch in downsample_in_out:
            self.down_blks.append(
                BasicDownBlock(in_ch, out_ch, norm=self.norm, block_num=self.block_num, kernel_size=5 if self.block_num >= 2 else 3)
            )
        
        if self.sam2_multi_scale_feates:
            self.mid_layer = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(32, 32, 1, 1), 
                    self.norm(32),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 64, 1, 1), 
                    self.norm(64),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, 1, 1), 
                    self.norm(256),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 256, 3, 1, 1), 
                    self.norm(256),
                    nn.GELU(),
                ),
            ])
        else:
            self.mid_layer = nn.Sequential(
                nn.Conv2d(downsample_in_out[-1][1] * 2, downsample_in_out[-1][1], 1, 1), 
                self.norm(downsample_in_out[-1][1]),
                nn.GELU(),
            )

        self.up_blks = nn.ModuleList()
        for _, (in_ch, out_ch) in enumerate(upsample_in_out):
            if skip_connect == 'concat':
                self.up_blks.append(BasicUpBlock(in_ch * 2, out_ch, skip_connect=skip_connect, norm=self.norm, block_num=self.block_num))
            else:
                self.up_blks.append(BasicUpBlock(in_ch, out_ch, skip_connect=skip_connect, norm=self.norm, block_num=self.block_num))

        self.matting_head = nn.Conv2d(matting_head_in, 1, 3, 1, 1)

        if self.norm_mask_logits == 'BN':
            self.logits_norm = self.norm(1)


    def preprocess_inputs(self, images, hq_features, pred_trimap):

        if self.wo_hq_features:
            return images

        if isinstance(self.norm_mask_logits, float):
            norm_hq_features = hq_features / self.norm_mask_logits
        elif self.norm_mask_logits == 'BN':
            norm_hq_features = self.logits_norm(hq_features)
        elif self.norm_mask_logits == 'Sigmoid':
            if hq_features.shape[1] == 1:
                norm_hq_features = torch.sigmoid(hq_features)
            else:
                norm_hq_features = torch.softmax(hq_features, dim=1)
        elif self.norm_mask_logits:
            norm_hq_features = hq_features / torch.std(hq_features, dim=(1, 2, 3), keepdim=True)
        else:
            norm_hq_features = hq_features

        if self.concat_gen_trimap:
            pred_trimap = F.interpolate(pred_trimap, size=(images.shape[-2], images.shape[-1]), mode='bilinear', align_corners=False)
            pred_trimap = torch.argmax(pred_trimap, dim=1, keepdim=True).float() / 2.0
            norm_hq_features = torch.concat((norm_hq_features, pred_trimap.detach()), dim=1)
        elif self.with_trimap:
            mask = (norm_hq_features > 0).float()
            for i_batch in range(images.shape[0]):
                mask_area = torch.sum(mask[i_batch])
                kernel_size = max(self.min_kernel_size, int((mask_area ** 0.5) / self.kernel_div))
                kernel_size = min(kernel_size, self.gen_trimap.max_kernal - 1)
                mask[i_batch, 0] = self.gen_trimap(mask[i_batch, 0], kernel_size=kernel_size)
            trimaps = mask
            norm_hq_features = torch.concat((norm_hq_features, trimaps), dim=1)

        conditional_images = torch.concatenate((images, norm_hq_features), dim=1)
        return conditional_images

    def forward(self, images, hq_features, vit_intern_feat, return_alpha_logits=False, pred_trimap=None):
        
        condition_input = self.preprocess_inputs(images, hq_features, pred_trimap)

        if not self.sam2_multi_scale_feates:
            # aggregate 4 vit_intern_feat
            # assert len(vit_intern_feat) == self.vit_intern_feat_num
            vit_feats = []
            for idx in self.vit_intern_feat_index:
                vit_feats.append(self.vit_feat_proj[str(idx)](vit_intern_feat[idx].permute(0, 3, 1, 2)))
            vit_feats = torch.concat(vit_feats, dim=1)
            vit_aggregation_feats = self.vit_feat_aggregation(vit_feats)

        details = []
        dfeatures = condition_input

        if hasattr(self, 'big_kernel_process'):
            dfeatures = self.big_kernel_process(dfeatures)

        for i in range(len(self.down_blks)):
            if self.sam2_multi_scale_feates:
                if i == 2:
                    dfeatures = torch.concat((dfeatures, self.mid_layer[0](vit_intern_feat['high_res_feats'][0])), dim=1)
                elif i == 3:
                    dfeatures = torch.concat((dfeatures, self.mid_layer[1](vit_intern_feat['high_res_feats'][1])), dim=1)
            dfeatures = self.down_blks[i](dfeatures)
            details.append(dfeatures)

        if self.sam2_multi_scale_feates:
            out = torch.concat((details[-1], self.mid_layer[2](vit_intern_feat['image_embed'])), dim=1)
            out = self.mid_layer[3](out)
        else:
            out = self.mid_layer(torch.concat((details[-1], vit_aggregation_feats), dim=1))
        for i in range(len(self.up_blks)):
            out = self.up_blks[i](out, details[-i - 1])
        alpha = torch.sigmoid(self.matting_head(out))
        if return_alpha_logits:
            return alpha, out
        else:
            return alpha



if __name__ == '__main__':

    from engine.mattingtrainer import parameter_count_table

    model = MattingDetailDecoder(img_feat_in = 5, vit_intern_feat_index=[0])
    x = torch.randn((2, 5, 1024, 1024))
    hq_features = torch.randn((2, 1, 1024, 1024))
    vit_feat = [torch.randn((2, 64, 64, 1024)) for _ in range(4)]

    out = model(x, hq_features, vit_feat)
    print(out.shape)

    print("Trainable parameters: \n" + parameter_count_table(model, trainable_only=True, max_depth=5))
