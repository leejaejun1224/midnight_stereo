# -*- coding: utf-8 -*-
# Improved Selective-IGEV (cosine-seeded + ACV-style gating + normalized GWC)
# Drop-in replacement for your decoder_selective_igev*.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Basic 2D/3D blocks & utils
# =========================
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
            x = nn.LeakyReLU(inplace=True)(x)
        return x

class Conv2x(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False,
                 concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
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
            x = nn.LeakyReLU(inplace=True)(x)
        return x

class Conv2x_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False,
                 concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
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
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    return volume.contiguous()

def disparity_regression(x, maxdisp):
    """x: [B,D,H,W], returns [B,1,H,W] with disparity in [0..maxdisp]"""
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
        return torch.sigmoid(feat_att) * cv

def context_upsample(disp_low, up_weights):
    # disp_low: (B,1,h,w), up_weights: (B,9,4h,4w)
    b, c, h, w = disp_low.shape
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)
    return (disp_unfold*up_weights).sum(1)

# =========================
# Encoders / Update blocks
# =========================
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
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes); self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes); self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)
        else:
            self.norm1 = nn.Sequential(); self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()
        self.downsample = None if (stride == 1 and in_planes == planes) else nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)

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

# =========================
# Stereo utils
# =========================
def bilinear_sampler(img, coords, mode='bilinear', mask=False):
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
        for _ in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)
        for _ in range(self.num_levels-1):
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

# =========================
# Update (GRU)
# =========================
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
        return self.sigmoid(self.samconv(torch.cat([avg_out, max_out], dim=1)))

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

# =========================
# Cost aggregation (3D hourglass)
# =========================
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
        return self.conv1_up(conv1)

# =========================
# NEW: Cosine-corr seed & ACV-style gater
# =========================
class CosCorrVolume1_4(nn.Module):
    """ BHWC L2-normalized feats -> [B,D,H4,W4] """
    def __init__(self, max_disp_px: int):
        super().__init__()
        self.D = max(1, max_disp_px // 4)

    @staticmethod
    def _shift_right_bchw(x: torch.Tensor, d: int) -> torch.Tensor:
        if d == 0: return x
        return F.pad(x, (d, 0, 0, 0))[:, :, :, :x.shape[-1]]

    def forward(self, L_bhwc: torch.Tensor, R_bhwc: torch.Tensor) -> torch.Tensor:
        # L_bhwc, R_bhwc: [B,H4,W4,C], already L2-normalized
        L = L_bhwc.permute(0, 3, 1, 2).contiguous()  # [B,C,H4,W4]
        R = R_bhwc.permute(0, 3, 1, 2).contiguous()
        vols = []
        for d in range(self.D):
            R_shift = self._shift_right_bchw(R, d)
            vols.append((L * R_shift).sum(dim=1, keepdim=False))  # [B,H4,W4]
        return torch.stack(vols, dim=1)  # [B,D,H4,W4]

class CorrToGWCGater(nn.Module):
    """ corr[B,D,H,W] -> weights[B,G,D,H,W] to gate GWC channels """
    def __init__(self, out_groups: int, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv3d(hidden, out_groups, 3, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, corr: torch.Tensor) -> torch.Tensor:
        return self.net(corr.unsqueeze(1))  # [B,G,D,H,W]

# =========================
# IGEV Stereo (improved)
# =========================
class IGEVStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims

        # context encoder & GRU update
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims],
                                      norm_fn="group", downsample=args.n_downsample)
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(args.hidden_dims[0])

        # image stems (2x / 4x)
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU(inplace=True))
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU(inplace=True))

        # SPX (upsample mask)
        self.spx   = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU(inplace=True))
        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru   = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        # match stem (1/4 → 96)
        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        # correlation stem / aggregation (keep channels=8)
        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        # ViTish 1/4 feature adapters (use fused_1_4 as "vit")
        vitc = getattr(args, 'fused_ch', None)
        if vitc is not None:
            self.vit_proj_1_4_to48 = nn.Conv2d(vitc, 48, 1, bias=False)
            self.vit_proj_8_to64   = nn.Conv2d(vitc, 64, 1, bias=False)
            self.vit_proj_16_to192 = nn.Conv2d(vitc, 192, 1, bias=False)
            self.vit_proj_32_to160 = nn.Conv2d(vitc, 160, 1, bias=False)

        # === NEW: cosine corr seed + GWC gater ===
        self.use_corr_seed = True
        self.seed_tau = float(getattr(args, "corr_seed_tau", 0.15))        # temperature
        self.seed_blend_alpha = float(getattr(args, "corr_seed_alpha", 1.0))
        self.num_gwc_groups = 8
        self.cos_corr = CosCorrVolume1_4(max_disp_px=args.max_disp_px)
        self.gwc_gater = CorrToGWCGater(out_groups=self.num_gwc_groups)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx); spx_pred = F.softmax(spx_pred, 1)
        return context_upsample(disp*4., spx_pred).unsqueeze(1)

    def _build_feats_from_vit(self, vit_1_4, stem_4x):
        """ViTish 1/4 -> [x4(96), x8(64), x16(192), x32(160)]"""
        f4_v = self.vit_proj_1_4_to48(vit_1_4)                 # [B,48,H/4,W/4]
        x4   = torch.cat([f4_v, stem_4x], dim=1)               # [B,96,H/4,W/4]
        f8  = F.avg_pool2d(vit_1_4, 2, 2)
        f16 = F.avg_pool2d(f8,     2, 2)
        f32 = F.avg_pool2d(f16,    2, 2)
        x8  = self.vit_proj_8_to64(f8)                         # [B,64 ,H/8 ,W/8 ]
        x16 = self.vit_proj_16_to192(f16)                      # [B,192,H/16,W/16]
        x32 = self.vit_proj_32_to160(f32)                      # [B,160,H/32,W/32]
        return [x4, x8, x16, x32]

    def forward(self, image1, image2,
                vit_left_1_4=None, vit_right_1_4=None,
                # NEW: BHWC cosine features from StereoModel (already L2-normalized)
                corr_left_bhwc=None, corr_right_bhwc=None,
                iters=12, flow_init=None, test_mode=False):

        # stems
        stem_2x = self.stem_2(image1); stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(image2); stem_4y = self.stem_4(stem_2y)

        # features pyramid
        assert (vit_left_1_4 is not None) and (vit_right_1_4 is not None), \
            "Pass StereoModel['fused_1_4'] as vit_left_1_4 / vit_right_1_4"
        features_left  = self._build_feats_from_vit (vit_left_1_4,  stem_4x)
        features_right = self._build_feats_from_vit (vit_right_1_4, stem_4y)

        # match features @1/4 (=== CHG: normalize to boost SNR ===)
        match_left  = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))
        match_left  = F.normalize(match_left,  dim=1)
        match_right = F.normalize(match_right, dim=1)

        # cosine correlation volume (BHWC → corr [B,D,H4,W4])
        if (corr_left_bhwc is not None) and (corr_right_bhwc is not None):
            corr_vol = self.cos_corr(corr_left_bhwc, corr_right_bhwc)
        else:
            # fallback: use normalized match features (approx cosine)
            L_bhwc = F.normalize(match_left.permute(0,2,3,1).contiguous(),  dim=-1)
            R_bhwc = F.normalize(match_right.permute(0,2,3,1).contiguous(), dim=-1)
            corr_vol = self.cos_corr(L_bhwc, R_bhwc)

        # GWC volume + (=== NEW) ACV-style gating with corr ===
        D_cells = int(self.args.max_disp_px // 4)
        gwc_volume = build_gwc_volume(match_left, match_right, D_cells, self.num_gwc_groups)  # [B,G,D,H,W]
        gwc_gate   = self.gwc_gater(corr_vol)                                                # [B,G,D,H,W]
        gwc_volume = gwc_volume * gwc_gate
        gwc_volume = self.corr_stem(gwc_volume)
        gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])

        # 3D aggregation
        geo_encoding_volume = self.cost_agg(gwc_volume, features_left)    # [B,8,D?,H,W]
        logits = self.classifier(geo_encoding_volume).squeeze(1)          # [B,D?,H,W]

        # === NEW: seed from cosine corr (softmax(corr/τ) → soft-argmin) ===
        with torch.no_grad():
            prob_seed = F.softmax(corr_vol / max(1e-6, self.seed_tau), dim=1)   # [B,Ds,H,W]
            D_seed    = prob_seed.size(1) - 1                                   # === CHG: dynamic ===
            disp_seed_cell = disparity_regression(prob_seed, D_seed)            # [B,1,H,W]
        disp_seed_px = disp_seed_cell * 4.0

        # learned initial (from logits)
        prob_learned = F.softmax(logits, dim=1)                                 # [B,Dl,H,W]
        D_learned    = prob_learned.size(1) - 1                                 # === CHG: dynamic ===
        disp_learned_px = disparity_regression(prob_learned, D_learned) * 4.0

        # blend (keep internal disp in cell units for GRU)
        alpha = float(self.seed_blend_alpha)
        disp = alpha * (disp_seed_px / 4.0) + (1.0 - alpha) * (disp_learned_px / 4.0)  # [B,1,H/4,W/4], cell units

        # SPX + context encoders
        xspx = self.spx_4(features_left[0]); xspx = self.spx_2(xspx, stem_2x)
        spx_pred = F.softmax(self.spx(xspx), 1)

        cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [self.cam(x) * x for x in inp_list]
        att = [self.sam(x) for x in inp_list]

        # geo fn for GRU refinements
        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(),
                           radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w, device=match_left.device).float().reshape(1,1,w,1).repeat(b, h, 1, 1)

        for _ in range(iters):
            disp = disp.detach()                 # [B,1,H/4,W/4], cell units
            geo_feat = geo_fn(disp, coords)
            net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
            disp = disp + delta_disp

        disp_1_4_px = disp * 4.0
        disp_full = self.upsample_disp(disp, mask_feat_4, stem_2x)

        return {
            'disp_1_4':  disp_1_4_px,
            'disp_full': disp_full,
            'corr_volume_1_4': corr_vol,  # for debugging/monitoring
        }
