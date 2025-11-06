# -*- coding: utf-8 -*-
# Auto-generated (modified) Single-File Selective-IGEV
#
# Upstream: Windsrain/Selective-Stereo (MIT)
# This file is a plain concatenation of the Selective-IGEV 'core' modules,
# with IGEVStereo modified to accept external ViT 1/4 features and to return
# exactly two outputs: (disp_1_4_px, disp_full).
#
# ---- BEGIN MIT LICENSE (UPSTREAM) ----
# (라이선스 원문은 원저장소 LICENSE 참고)
# ---- END MIT LICENSE (UPSTREAM) ----

# ===== BEGIN core/submodule.py =====
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)
        return x

class Conv2x(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True,
                                   kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True,
                                   kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x

class BasicConv_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()
        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)
        return x

class Conv2x_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True,
                                      kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True,
                                      kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels*2, out_channels*mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x

def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def correlation(fea1, fea2):
    cost = torch.sum((fea1 * fea2), dim=1, keepdim=True)
    return cost

def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume

def disparity_regression(x, maxdisp):

    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp+1, dtype=x.dtype, device=x.device).view(1, maxdisp+1, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)

class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()
        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1))

    def forward(self, cv, feat):
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att)*cv
        return cv

def context_upsample(disp_low, up_weights):
    # disp_low: (B,1,h,w), up_weights: (B,9,4h,4w)
    b, c, h, w = disp_low.shape
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)
    disp = (disp_unfold*up_weights).sum(1)
    return disp
# ===== END core/submodule.py =====

# ===== BEGIN core/extractor.py =====
import torch
import torch.nn as nn
import torch.nn.functional as F
# (Flattened: no import from core.submodule; symbols are in global ns)
import timm

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes): self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes); self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes): self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes); self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes): self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential(); self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes): self.norm3 = nn.Sequential()
        self.downsample = None if (stride == 1 and in_planes == planes) else nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None: x = self.downsample(x)
        return self.relu(x+y)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, downsample=3):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn; self.downsample = downsample
        if norm_fn == 'group': self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif norm_fn == 'batch': self.norm1 = nn.BatchNorm2d(64)
        elif norm_fn == 'instance': self.norm1 = nn.InstanceNorm2d(64)
        else: self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96,  stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None: nn.init.constant_(m.weight, 1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x, dual_inp=False):
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]; x = torch.cat(x, dim=0)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.conv2(x)
        if self.training and self.dropout is not None: x = self.dropout(x)
        if is_list: x = x.split(split_size=batch_dim, dim=0)
        return x

class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn; self.downsample = downsample
        if norm_fn == 'group': self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif norm_fn == 'batch': self.norm1 = nn.BatchNorm2d(64)
        elif norm_fn == 'instance': self.norm1 = nn.InstanceNorm2d(64)
        else: self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)
        output_list = []
        for dim in output_dim:
            output_list.append(nn.Sequential(ResidualBlock(128, 128, self.norm_fn, stride=1),
                                             nn.Conv2d(128, dim[2], 3, padding=1)))
        self.outputs04 = nn.ModuleList(output_list)
        output_list = []
        for dim in output_dim:
            output_list.append(nn.Sequential(ResidualBlock(128, 128, self.norm_fn, stride=1),
                                             nn.Conv2d(128, dim[1], 3, padding=1)))
        self.outputs08 = nn.ModuleList(output_list)
        output_list = []
        for dim in output_dim:
            output_list.append(nn.Conv2d(128, dim[0], 3, padding=1))
        self.outputs16 = nn.ModuleList(output_list)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None: nn.init.constant_(m.weight, 1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x, dual_inp=False, num_layers=3):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        if dual_inp:
            v = x; x = x[:(x.shape[0]//2)]
        outputs04 = [f(x) for f in self.outputs04]
        if num_layers == 1: return (outputs04, v) if dual_inp else (outputs04,)
        y = self.layer4(x); outputs08 = [f(y) for f in self.outputs08]
        if num_layers == 2: return (outputs04, outputs08, v) if dual_inp else (outputs04, outputs08)
        z = self.layer5(y); outputs16 = [f(z) for f in self.outputs16]
        return (outputs04, outputs08, outputs16, v) if dual_inp else (outputs04, outputs08, outputs16)

class SubModule(nn.Module):
    def __init__(self): super(SubModule, self).__init__()
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels; m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.kernel_size[2]*m.out_channels; m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d): m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d): m.weight.data.fill_(1); m.bias.data.zero_()

class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained = True
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem; self.bn1 = model.bn1; self.act1 = model.act1
        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])
        self.deconv32_16 = Conv2x_IN(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8  = Conv2x_IN(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4   = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4       = BasicConv_IN(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        x16 = self.deconv32_16(x32, x16)
        x8  = self.deconv16_8(x16, x8)
        x4  = self.deconv8_4(x8, x4)
        x4  = self.conv4(x4)
        return [x4, x8, x16, x32]
# ===== END core/extractor.py =====

# ===== BEGIN core/utils/utils.py =====
import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]
    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))
    x1 = x0 + dx; y1 = y0 + dy
    x1 = x1.reshape(-1); y1 = y1.reshape(-1); dx = dx.reshape(-1); dy = dy.reshape(-1)
    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]; y1 = y1[valid]; dx = dx[valid]; dy = dy[valid]
    flow_x = interpolate.griddata((x1, y1), dx, (x0, y0), method='nearest', fill_value=0)
    flow_y = interpolate.griddata((x1, y1), dy, (x0, y0), method='nearest', fill_value=0)
    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    assert torch.unique(ygrid).numel() == 1 and H == 1  # stereo
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def gauss_blur(input, N=5, std=1):
    B, D, H, W = input.shape
    x, y = torch.meshgrid(torch.arange(N).float() - N//2, torch.arange(N).float() - N//2)
    unnormalized_gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * std ** 2))
    weights = unnormalized_gaussian / unnormalized_gaussian.sum().clamp(min=1e-4)
    weights = weights.view(1,1,N,N).to(input)
    output = F.conv2d(input.reshape(B*D,1,H,W), weights, padding=N//2)
    return output.view(B, D, H, W)
# ===== END core/utils/utils.py =====

# ===== BEGIN core/geometry.py =====
import torch
import torch.nn.functional as F
# bilinear_sampler from utils already in ns

class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)
        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)
        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)
        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=disp.device).view(1, 1, 2*r+1, 1)
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)
            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl).view(b, h, w, -1)
            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl).view(b, h, w, -1)
            out_pyramid.append(geo_volume); out_pyramid.append(init_corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr
# ===== END core/geometry.py =====

# ===== BEGIN core/update.py =====
import torch
import torch.nn as nn
import torch.nn.functional as F

class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.conv2(self.relu(self.conv1(x)))

class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(), nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x)); max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionExtractor, self).__init__()
        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.samconv(x))

class RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3, dilation=1):
        super(RaftConvGRU, self).__init__()
        pad = (kernel_size+(kernel_size-1)*(dilation-1))//2
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=pad, dilation=dilation)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=pad, dilation=dilation)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=pad, dilation=dilation)
    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx)); r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        return (1-z) * h + z * q

class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, small_kernel_size=1, large_kernel_size=3):
        super(SelectiveConvGRU, self).__init__()
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)
    def forward(self, att, h, *x):
        x = torch.cat(x, dim=1)
        return self.small_gru(h, x) * att + self.large_gru(h, x) * (1 - att)

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv   = nn.Conv2d(64+64, 128-1, 3, padding=1)
    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr)); cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp)); disp_ = F.relu(self.convd2(disp_))
        out = F.relu(self.conv(torch.cat([cor, disp_], dim=1)))
        return torch.cat([out, disp], dim=1)

def pool2x(x): return F.avg_pool2d(x, 3, stride=2, padding=1)
def interp(x, dest): return F.interpolate(x, dest.shape[2:], mode='bilinear', align_corners=True)

class BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128
        if args.n_gru_layers == 3:
            self.gru16 = SelectiveConvGRU(hidden_dims[0], hidden_dims[0] + hidden_dims[1])
        if args.n_gru_layers >= 2:
            self.gru08 = SelectiveConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[1] + hidden_dims[2])
        self.gru04 = SelectiveConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1) + hidden_dims[2])
        self.disp_head = DispHead(hidden_dims[2], 256)
        self.mask_feat_4 = nn.Sequential(nn.Conv2d(hidden_dims[2], 32, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, net, inp, corr, disp, att):
        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(att[2], net[2], inp[2], pool2x(net[1]))
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]))
        motion_features = self.encoder(disp, corr)
        motion_features = torch.cat([inp[0], motion_features], dim=1)
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(att[0], net[0], motion_features, interp(net[1], net[0]))
        delta_disp = self.disp_head(net[0])
        mask_feat_4 = .25 * self.mask_feat_4(net[0])  # scale to balance grads
        return net, mask_feat_4, delta_disp
# ===== END core/update.py =====

# ===== BEGIN core/igev_stereo.py (MODIFIED) =====
import torch
import torch.nn as nn
import torch.nn.functional as F
# update / extractor / geometry / submodule symbols already in ns

# try:
#     autocast = torch.cuda.amp.autocast
# except:
#     class autocast:
#         def __init__(self, enabled): pass
#         def __enter__(self): pass
#         def __exit__(self, *args): pass

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()
        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1))
        self.conv2 = nn.Sequential(
            BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1))
        self.conv3 = nn.Sequential(
            BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1))
        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True, relu=True,
                                  kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True, relu=True,
                                  kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False, relu=False,
                                  kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.agg_0 = nn.Sequential(
            BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1))
        self.agg_1 = nn.Sequential(
            BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))
        # Feature attention at multi-scales (expects feat_ch = 64/192/160)
        self.feature_att_8      = FeatureAtt(in_channels*2, 64)
        self.feature_att_16     = FeatureAtt(in_channels*4, 192)
        self.feature_att_32     = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16  = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8   = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x);  conv1 = self.feature_att_8(conv1,  features[1])
        conv2 = self.conv2(conv1); conv2 = self.feature_att_16(conv2, features[2])
        conv3 = self.conv3(conv2); conv3 = self.feature_att_32(conv3, features[3])
        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1); conv2 = self.agg_0(conv2); conv2 = self.feature_att_up_16(conv2, features[2])
        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1); conv1 = self.agg_1(conv1); conv1 = self.feature_att_up_8(conv1, features[1])
        conv = self.conv1_up(conv1)
        return conv

class IGEVStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims

        # context encoder & GRU update
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="group", downsample=args.n_downsample)
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(128)

        # (A) 기존 MobileNetV2 기반 Feature() - ViT 피처 미제공 시 사용
        # self.feature = Feature()

        # (B) 이미지 stem (2x, 4x) — 여전히 사용 (spx, match 보강)
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU())
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU())

        # SPX(초해상 업샘플 마스크 추정)
        self.spx   = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU())
        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru   = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        # match feature stem (1/4에서 96→96)
        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        # correlation stem & aggregation
        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        # === (NEW) ViT 1/4 feature adapter ===
        # args.vit_ch_1_4: ViT 1/4 feature channel count (e.g., 256/320)
        vitc = getattr(args, 'fused_ch', None)
        if vitc is not None:
            # 1/4 레벨: ViT( Cvit ) → 48ch 로 project 후 stem_4x(48)와 concat → 96ch
            self.vit_proj_1_4_to48 = nn.Conv2d(vitc, 48, 1, bias=False)
            # 상위 스케일 FeatureAtt 용 프로젝션 (채널 고정 규격)
            self.vit_proj_8_to64   = nn.Conv2d(vitc, 64, 1, bias=False)
            self.vit_proj_16_to192 = nn.Conv2d(vitc, 192, 1, bias=False)
            self.vit_proj_32_to160 = nn.Conv2d(vitc, 160, 1, bias=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        # with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx); spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)
        return up_disp

    def _build_feats_from_vit(self, vit_1_4, stem_4x):
        """ViT 1/4 -> [x4(96), x8(64), x16(192), x32(160)]"""
        # x4(1/4)
        f4_v = self.vit_proj_1_4_to48(vit_1_4)                 # [B,48,H/4,W/4]
        x4   = torch.cat([f4_v, stem_4x], dim=1)               # [B,96,H/4,W/4]
        # x8/x16/x32: avgpool + projection
        f8  = F.avg_pool2d(vit_1_4, 2, 2)                      # [B,Cvit,H/8,W/8]
        f16 = F.avg_pool2d(f8, 2, 2)                           # [B,Cvit,H/16,W/16]
        f32 = F.avg_pool2d(f16, 2, 2)                          # [B,Cvit,H/32,W/32]
        x8  = self.vit_proj_8_to64(f8)                         # [B,64 ,H/8 ,W/8 ]
        x16 = self.vit_proj_16_to192(f16)                      # [B,192,H/16,W/16]
        x32 = self.vit_proj_32_to160(f32)                      # [B,160,H/32,W/32]
        return [x4, x8, x16, x32]

    def forward(self, image1, image2, vit_left_1_4=None, vit_right_1_4=None, iters=12, flow_init=None, test_mode=False):
        """Estimate disparity between a stereo pair.

        Returns:
            disp_1_4_px: [B,1,H/4,W/4], disparity values in **pixels** (cell*4)
            disp_full  : [B,1,H,  W  ], full-resolution disparity in pixels
        """
        # image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        # image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        use_vit = (vit_left_1_4 is not None) and (vit_right_1_4 is not None) 

        # with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
        # stems (이미지 기반 경량 특징, 기존과 동일)
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)

        # 멀티스케일 features 준비
        if use_vit:
            # ViT 경로: 외부 1/4 피처를 피라미드로 어댑팅
            features_left  = self._build_feats_from_vit (vit_left_1_4,  stem_4x)   # [x4(96), x8(64), x16(192), x32(160)]
            features_right = self._build_feats_from_vit (vit_right_1_4, stem_4y)
        # else:
        #     # 기존 Feature() 경로 (MobileNetV2)
        #     features_left  = self.feature(image1)
        #     features_right = self.feature(image2)
        #     # 1/4에서 stem_4x를 붙여 96ch로 (기존 코드 유지)
        #     features_left[0]  = torch.cat((features_left[0],  stem_4x), 1)
        #     features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        # 매칭용 1/4 특징(96ch) -> 96ch desc
        match_left  = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        # GWC cost volume (1/4)
        gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp_px//4, 8)
        gwc_volume = self.corr_stem(gwc_volume)
        gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
        geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

        # Stage-0: 초기 disp (1/4 cell 단위)
        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        disp = disparity_regression(prob, self.args.max_disp_px//4)  # [B,1,H/4,W/4] in cell units
        del prob, gwc_volume

        # spx mask (train/eval 동일하게 계산해 사용)
        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        # context encoder for GRU
        cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [self.cam(x) * x for x in inp_list]
        att = [self.sam(x) for x in inp_list]

        # Combined Geo Encoding Volume function
        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(),
                           radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w, device=match_left.device).float().reshape(1,1,w,1).repeat(b, h, 1, 1)

        # GRU refinement loops (1/4 cell 단위로 업데이트)
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            # with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
            disp = disp + delta_disp

        # 최종 출력들
        # 1) 1/4 해상도 disparity (픽셀단위로 환산)
        disp_1_4_px = disp * 4.0

        # 2) 원본 해상도 disparity (convex upsampling)
        disp_full = self.upsample_disp(disp, mask_feat_4, stem_2x)

        # (참고) 초기 full-res도 원하면: init_full = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        pred = {
            'disp_1_4': disp_1_4_px,
            'disp_full': disp_full,
        }
        return pred
# ===== END core/igev_stereo.py =====
