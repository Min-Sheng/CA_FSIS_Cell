import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.autograd import Function

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import numpy as np

DEBUG = False

class GradReverse(Function):
    '''
        Gradient reversal layer
    '''

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=-1):
    return GradReverse.apply(x, alpha)


class domain_discriminator_im(nn.Module):
    """ Image-level adversarial domain classifier """
    def __init__(self, n_in, n_out=14):
        super(domain_discriminator_im, self).__init__()
        self.conv_1 = nn.Conv2d(n_in, 512, kernel_size=1, padding=0, stride=1)
        #self.conv_2 = nn.Conv2d(512, 1, kernel_size=1, padding=0, stride=1)
        self.conv_2 = nn.Conv2d(512, n_out, kernel_size=1, padding=0, stride=1)
        self.leaky_relu  = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, alpha=-1):
        img_features = []
        for feature in x:
            feat = grad_reverse(feature, alpha)
            feat = self.conv_1(feat)
            feat = self.leaky_relu (feat)
            feat = self.conv_2(feat)
            img_features.append(feat)
        return img_features

    def detectron_weight_mapping(self):
        mapping = {}
        orphan_in_detectron = []
        return mapping, orphan_in_detectron


def domain_loss_im(pred, domain_label):
    """
    Image-level domain adversarial loss
    
    """
    pred_flattened = []
    target_label_flattened = []
    for pred_per_level in pred:
        N, A, H, W = pred_per_level.shape
        #target_label = domain_label.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, A, H, W)
        target_label = domain_label.unsqueeze(1).unsqueeze(1).repeat(1, H, W)
        #pred_per_level = pred_per_level.reshape(N, -1)
        pred_per_level = pred_per_level.permute(0, 2, 3, 1).contiguous().reshape(N, -1, A)
        target_label = target_label.reshape(N, -1)
        pred_flattened.append(pred_per_level)
        target_label_flattened.append(target_label)

    #pred_flattened = torch.cat(pred_flattened, dim=1).reshape(N, -1)
    #target_label_flattened = torch.cat(target_label_flattened, dim=1).reshape(N, -1)
    #loss_da_im = F.binary_cross_entropy_with_logits(pred_flattened, target_label_flattened.float())
    pred_flattened = torch.cat(pred_flattened, dim=1).reshape(-1, A)
    target_label_flattened = torch.cat(target_label_flattened, dim=1).reshape(-1)
    loss_da_im = F.nll_loss(torch.log(F.softmax(pred_flattened, dim=1) + 1e-3), target_label_flattened.long())
    
    return loss_da_im


class domain_discriminator_roi(nn.Module):
    """ ROI-level adversarial domain classifier """
    def __init__(self, n_in, roi_xform_func, spatial_scale, n_out=14):
        super(domain_discriminator_roi, self).__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.conv_1 = nn.Conv2d(n_in, 1024, kernel_size=1, stride=1)
        self.conv_2  = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.conv_3 = nn.Conv2d(1024, n_out, kernel_size=1, stride=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, rpn_ret, alpha=-1):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        x = grad_reverse(x, alpha)
        x = self.conv_1(x)
        x = self.leaky_relu(x)

        x = self.conv_2(x)
        x = self.leaky_relu(x)

        x = self.conv_3(x)

        return x

    def detectron_weight_mapping(self):
        mapping = {}
        orphan_in_detectron = []
        return mapping, orphan_in_detectron


def domain_loss_roi(pred, domain_label, rois):
    """
    ROI-level domain adversarial loss
    """
    device_id = pred.get_device()
    rois = Variable(torch.from_numpy(rois[:,0].astype('int64'))).cuda(device_id)
    domain_label_expanded = domain_label[rois]
    N, A, H, W = pred.shape
    target_label = domain_label_expanded.unsqueeze(1).unsqueeze(1).repeat(1, H, W)
    pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1, A)
    target_label = target_label.reshape(-1)
    #loss_da_roi = F.binary_cross_entropy_with_logits(pred, target_label.float())
    loss_da_roi = F.nll_loss(torch.log(F.softmax(pred, dim=1) + 1e-3), target_label.long())
    
    return loss_da_roi


def domain_loss_cst(im_pred, roi_pred, rois):
    """
    Consistency regularization between image and ROI predictions
    
    """
    device_id = roi_pred.get_device()
    roi_pred = roi_pred.mean(3).mean(2)
    rois = Variable(torch.from_numpy(rois[:,0].astype('int64'))).cuda(device_id)
    
    loss_cst = []
    for im_pred_per_level in im_pred:
        assert im_pred_per_level.get_device() == roi_pred.get_device()
        N, A, H, W = im_pred_per_level.shape
        im_pred_mean_per_level = im_pred_per_level.mean(3).mean(2)
        im_pred_mean_expanded = im_pred_mean_per_level[rois]
        loss_cst.append(torch.mean((im_pred_mean_expanded - roi_pred)**2).unsqueeze(0))
    
    loss_cst = torch.cat(loss_cst, dim=0).mean()
    
    return loss_cst