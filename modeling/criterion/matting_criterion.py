import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class MattingCriterion(nn.Module):
    def __init__(
        self,
        *,
        losses,
        image_size = 1024,
    ):
        super(MattingCriterion, self).__init__()
        self.losses = losses
        self.image_size = image_size

    def loss_gradient_penalty(self, sample_map, preds, targets):

        #sample_map for unknown area
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            scale = sample_map.shape[0] * (self.image_size ** 2) / torch.sum(sample_map)

        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        #gradient in y 
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        #loss
        loss = (F.l1_loss(delta_pred_x * sample_map, delta_gt_x * sample_map) * scale + \
            F.l1_loss(delta_pred_y * sample_map, delta_gt_y * sample_map) * scale + \
            0.01 * torch.mean(torch.abs(delta_pred_x * sample_map)) * scale +  \
            0.01 * torch.mean(torch.abs(delta_pred_y * sample_map)) * scale)

        return dict(loss_gradient_penalty=loss)

    def loss_pha_laplacian(self, preds, targets):
        loss = laplacian_loss(preds, targets)
        return dict(loss_pha_laplacian=loss)

    def unknown_l1_loss(self, sample_map, preds, targets):
        
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            scale = sample_map.shape[0] * (self.image_size ** 2) / torch.sum(sample_map)
        # scale = 1

        loss = F.l1_loss(preds * sample_map, targets * sample_map) * scale

        return dict(unknown_l1_loss=loss)

    def known_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0] * (self.image_size ** 2) / torch.sum(new_sample_map)
        # scale = 1
        
        loss = F.l1_loss(preds * new_sample_map, targets * new_sample_map) * scale

        return dict(known_l1_loss=loss)

    def get_loss(self, k, sample_map, preds, targets):
        if k=='unknown_l1_loss' or k=='known_l1_loss' or k=='loss_gradient_penalty':
            losses = getattr(self, k)(sample_map, preds, targets)
        else:
            losses = getattr(self, k)(preds, targets)
        assert len(list(losses.keys())) == 1
        return losses[list(losses.keys())[0]]

    def forward(self, sample_map, preds, targets, batch_weight=None):
        losses = {i: torch.tensor(0.0, device=sample_map.device) for i in self.losses}
        for k in self.losses:
            if batch_weight is None:
                losses[k] += self.get_loss(k, sample_map, preds, targets)
            else:
                for i, loss_weight in enumerate(batch_weight):
                    if loss_weight == -1.0 and k != 'known_l1_loss':
                        continue
                    else:
                        losses[k] += self.get_loss(k, sample_map[i: i + 1], preds[i: i + 1], targets[i: i + 1]) * abs(loss_weight)
        return losses


#-----------------Laplacian Loss-------------------------#
def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]

def normalized_focal_loss(pred, gt, gamma=2, class_num=3, norm=True, beta_detach=False, beta_sum_detach=False):
    pred_logits = F.softmax(pred, dim=1)  # [B, 3, H, W]
    gt_one_hot = F.one_hot(gt, class_num).permute(0, 3, 1, 2)  # [B, 3, H, W]
    p = (pred_logits * gt_one_hot).sum(dim=1)  # [B, H, W]
    beta = (1 - p) ** gamma  # [B, H, W]
    beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True) / (pred.shape[-1] * pred.shape[-2])  # [B, 1, 1]

    if beta_detach:
        beta = beta.detach()
    if beta_sum_detach:
        beta_sum = beta_sum.detach()

    if norm:
        loss = 1 / beta_sum * beta * (-torch.log(p))
        return torch.mean(loss)
    else:
        loss = beta * (-torch.log(p))
        return torch.mean(loss)

class GHMC(nn.Module):
    def __init__(self, bins=10, momentum=0.75, loss_weight=1.0, device='cuda', norm=False):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.loss_weight = loss_weight
        self.device = device
        self.norm = norm

    def forward(self, pred, target, *args, **kwargs):
        """Calculate the GHM-C loss.
        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """

        # the target should be binary class label
        # if pred.dim() != target.dim():
        #     target, label_weight = _expand_binary_labels(
        #                             target, label_weight, pred.size(-1))
        # target, label_weight = target.float(), label_weight.float()
        # pdb.set_trace()

        # pred: [B, C, H, W], target: [B, H, W]
        pred = pred.permute(0, 2, 3, 1).reshape(-1, 3)  # [B x H x W, C]
        target = target.reshape(-1)  # [B x H x W]
        # self.acc_sum = self.acc_sum.type(pred.dtype)

        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros((target.shape),dtype=pred.dtype).to(self.device)

        # gradient length
        #g = 1 - torch.index_select(F.softmax(pred,dim=1).detach(), dim=0, index=target)
        g = 1 - torch.gather(F.softmax(pred,dim=1).detach(),dim=1,index=target.unsqueeze(1))
        #g = torch.abs(pred.softmax(2).detach() - target)

        tot = 1.0
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                idx = torch.nonzero(inds)[:, 0]
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    # pdb.set_trace()#scatter_ index_put_
                    #BB=torch.nonzero(inds)
                    _weight_idx = tot / self.acc_sum[i]
                    weights = weights.to(dtype=_weight_idx.dtype)
                    weights[idx] = _weight_idx
                    # weights.scatter_(0, torch.nonzero(inds)[:,0], tot / self.acc_sum[i])
                    # # weights.index_put_(inds, tot / self.acc_sum[i])
                    # weights[inds] = tot / self.acc_sum[i] # * torch.ones((len(inds)))
                else:
                    weights[idx] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

            # pdb.set_trace()
            # loss = (weights * F.cross_entropy(pred, target, reduction='none')).sum() / tot / pred.shape[0]
        if self.norm:
            weights = weights / torch.sum(weights).detach()

        loss = - ((weights.unsqueeze(1) * torch.gather(F.log_softmax(pred, dim=1), dim=1, index=target.unsqueeze(1))).sum() )  # / pred.shape[0]

        # loss3= F.cross_entropy(pred, target, reduction='mean')
        # loss4 = - ((torch.gather(F.log_softmax(pred, dim=1), dim=1, index=target.unsqueeze(1))).sum() / pred.shape[0])

        # pro = F.softmax(logits, dim=1)
        #
        # label_onehot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
        # with torch.no_grad():
        #     weight_matrix = (1 - pro) ** self.gamma
        # # pdb.set_trace()
        # fl = - (weight_matrix * (label_onehot * (pro + self.eps).log())).sum() / pro.shape[0]

        return loss

if __name__ == '__main__':
    pred = torch.randn(2, 3, 1024, 1024)
    gt =torch.argmax(torch.randn(2, 3, 1024, 1024), dim=1)
    loss = normalized_focal_loss(pred, gt)
    print(loss)
    


