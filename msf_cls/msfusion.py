# Date: 18/04/2022
# PyTorch implementation of MSFusionNet
# Author: Wilson Zhang


import torch

import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


def obtain_normalization(choice: int = 0, channels: int = 0, groups: int = 0):
    # 0 - Batch Normalization
    # 1 - Instance Normalization
    # 2 - Group Normalization
    # 3 - Layer Normalization
    if not choice:
        return nn.BatchNorm2d(channels)
    elif not choice - 1:
        return nn.InstanceNorm2d(channels)
    elif not choice - 2:
        return nn.GroupNorm(groups, channels)
    elif not choice - 3:
        return nn.GroupNorm(1, channels)  # still layer norm
    else:
        raise NotImplementedError('Unknown choice for normalization layer is indicated.')


class Up(nn.Module):
    """
    Update on 26/10/2020:
    When bilinear=True & conv=None, out_channels param will be invalid.

    Up-scale the first input, then concatenate with the second input and pass into self.conv (if specified).
    """

    def __init__(self, in_channels, out_channels, conv=None, bilinear=False, shortcut_channels=0, **kwargs):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        bias = kwargs.get('bias', True)
        shortcut_channels = out_channels if not shortcut_channels else shortcut_channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels >> 1, kernel_size=2, stride=2, bias=bias)
            in_channels >>= 1
        if conv is not None:
            self.conv = conv(in_channels + shortcut_channels, out_channels,
                             **kwargs)
        else:
            self.conv = conv
        # self.bn = obtain_normalization(0, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is BCHW
        diff_x = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x >> 1, diff_x - (diff_x >> 1),
                        diff_y >> 1, diff_y - (diff_y >> 1)])
        x = torch.cat([x2, x1], dim=1)
        if self.conv is not None:
            x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    """
    (3*3 convolution => [Norm] => Non-linear) * 2
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 mid_channels=None, normalization: int = 0, leaky_relu=False, bias=True, **kwargs):
        super(DoubleConv, self).__init__()
        mid_channels = mid_channels if mid_channels else out_channels
        groups = 32 if not mid_channels % 32 else mid_channels
        groups_out = 32 if not out_channels % 32 else out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            # following He's paper, group num = 32.
            obtain_normalization(choice=normalization, channels=mid_channels, groups=groups),
            nn.LeakyReLU(inplace=True) if leaky_relu else nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            obtain_normalization(choice=normalization, channels=out_channels, groups=groups_out),
            nn.LeakyReLU(inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class cls_head(nn.Module):
    def __init__(self, input_c, kernel_num, output_c):
        super(cls_head, self).__init__()
        self.conv = OutConv(in_channels=input_c * kernel_num << 4, out_channels=output_c)
        self.linear = nn.Sequential(
            nn.Linear(output_c * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_c)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #  x -> [2, 4(1), 1024, 14, 14]
        x = x.transpose(0, 1)
        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.conv(x)
        # x -> [4(1), 2, 14, 14]
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        res = self.softmax(x)
        return res


class MSFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, branch_num, kernel_size=3, padding=1,
                 normalization: int = 0, leaky_relu=False, pooling=False, bias=True):
        """
        API for Vanilla MSFusion Block, the backbone for each branch is DoubleConv following U-Net.
        Please note that pooling can be applied to output features within this block by specifying the pooling param,
        which will cause the output feature map has lower resolution.
        :param in_channels: number of input channels for each branch
        :param out_channels: number of output channels for each branch
        :param branch_num: number of branches
        :param kernel_size: kernel size for conv layers
        :param padding: padding size for conv layers
        :param normalization: choice for normalization layers
        :param leaky_relu: whether to replace relu layers with leaky relu layers.
        :param pooling: whether to adopt max pooling layer after encoder.
        :param bias: whether to preserve bias in conv layers.
        """
        super(MSFusionBlock, self).__init__()
        encoder = []
        self.pooling = nn.MaxPool2d(2) if pooling else None
        backbone = DoubleConv
        for _ in range(branch_num):
            encoder.append(backbone(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                    normalization=normalization, leaky_relu=leaky_relu, bias=bias))
        self.encoders = nn.ModuleList([encoder.pop() for _ in range(branch_num)])
        self.alphas = nn.Parameter(torch.randn(branch_num, requires_grad=True))

    def forward(self, x: torch.Tensor):
        # each input should follow the branch_numxBxCx(Nx)HxW shape.
        encoder_outputs = []
        # x_tensor = torch.tensor(x)
        for idx, branch in enumerate(self.encoders):
            encoder_outputs.append(branch(x[idx]))  # the conv output after encoder

        conv_maps = torch.Tensor()
        conv_maps = conv_maps.cuda('cuda:%s' % x.get_device()) if x.is_cuda else conv_maps.cpu()
        for encoder_output in encoder_outputs:
            conv_maps = torch.cat([conv_maps, torch.mean(encoder_output, dim=1, keepdim=True).unsqueeze(0)], dim=0)
            # same as average_pooling
            # conv_maps should follows the shape of branch_numxBx1x(Nx)HxW
        f_map = torch.sum(conv_maps, dim=0)  # corresponds to map1, shape should be B1(N)HW
        dot_products = torch.Tensor()
        dot_products = dot_products.cuda('cuda:%s' % x.get_device()) if x.is_cuda else dot_products.cpu()
        for idx in range(conv_maps.shape[0]):
            if x[idx].shape[1] - 1:
                v = f_map * torch.mean(x[idx], dim=1, keepdim=True)
                # same as multi_map, shape should be B1(N)HW
            else:
                v = f_map * x[idx]
            dot_products = torch.cat([dot_products,
                                      v.unsqueeze(0) * self.alphas[idx].exp() / (self.alphas.exp().sum())],
                                     dim=0)
            # same as m3_MultiLayer

        for idx in range(len(encoder_outputs)):
            encoder_outputs[idx] = encoder_outputs[idx] + dot_products[idx]
            # same as add_map.
        if self.pooling is not None:
            output = torch.cat([self.pooling(encoder_output).unsqueeze(0) for encoder_output in encoder_outputs], dim=0)
        else:
            output = torch.cat([encoder_output.unsqueeze(0) for encoder_output in encoder_outputs], dim=0)
        return output


class MSFusionNet(nn.Module):
    def __init__(self, input_c, output_c, kernel_num=64, normalization=0, leaky_relu=False,
                 bayesian=False, p=0.2, task='seg', **kwargs):
        """
        Update on 12/04/2021:
        MSFusionNet for prostate cancer segmentation.
        :param input_c: channel number of input data.
        :param output_c: channel number of output data.
        :param kernel_num: number of kernels for convolution layers in the first stage. For the following stages, the
        number of kernels will be 2 * n * kernel_num, where n denotes the depth of current stage.
        :param kernel_size (int or tuple): kernel size for convolution, pooling, up-convolution layers.
        :param normalization (int): choice of the normalization layer， default is 0.
        0 - batch normalization;
        1 - instance normalization;
        2 - group normalization.
        :param leaky_relu (bool): whether to replace relu layers with leaky_relu, default is False.
        :param device_ids (list): Specify the gpu ids used for parallel training. Currently our forward function can
        distribute the model to up to four GPUs.
        :param bayesian (bool): whether to apply dropout2d to the encoded and decoded features.
        :param p (float): dropout rate for all inserted dropout layers.
        """
        super(MSFusionNet, self).__init__()

        non_linear = nn.LeakyReLU if leaky_relu else nn.ReLU
        self.name = 'msf'
        """
        Some name problem with the input channel, it seems the parameter input_c denotes the branch number actually
        Here I set n_channels equals to 1 manually.
        """
        self.n_channels = 1
        self.n_classes = output_c
        self.bilinear = True
        self.task = task
        assert task in ['seg', 'cls']
        print("specified for {}".format("segmentation" if task == 'seg' else "classification"))
        # pooling is performed between stages manually to make decoding process easier.
        self.inc = MSFusionBlock(1, kernel_num, input_c, kernel_size=3, padding=1,
                                 normalization=normalization, leaky_relu=leaky_relu)
        self.encoder1 = MSFusionBlock(kernel_num, kernel_num << 1, input_c, kernel_size=3, padding=1,
                                      normalization=normalization, leaky_relu=leaky_relu)
        self.encoder2 = MSFusionBlock(kernel_num << 1, kernel_num << 2, input_c, kernel_size=3, padding=1,
                                      normalization=normalization, leaky_relu=leaky_relu)
        self.encoder3 = MSFusionBlock(kernel_num << 2, kernel_num << 3, input_c, kernel_size=3, padding=1,
                                      normalization=normalization, leaky_relu=leaky_relu)
        self.encoder4 = MSFusionBlock(kernel_num << 3, kernel_num << 4, input_c, kernel_size=3, padding=1,
                                      normalization=normalization, leaky_relu=leaky_relu)
        self.pooling = nn.MaxPool2d(2)
<<<<<<< HEAD
=======
        if self.task == 'cls':
            self.cls_head = cls_head(input_c, kernel_num, output_c)
>>>>>>> eb72c49a3d11a3f74e4981b6de00775f57a04b4d
        kernel_num <<= 4

        self.decoder1 = nn.ModuleList([
            Up(kernel_num, kernel_num >> 1, DoubleConv,
               normalization=normalization, leaky_relu=leaky_relu) for _ in range(input_c)
        ])

        self.decoder2 = nn.ModuleList([
            Up(kernel_num >> 1, kernel_num >> 2, DoubleConv,
               normalization=normalization, leaky_relu=leaky_relu) for _ in range(input_c)
        ])
        self.decoder3 = nn.ModuleList([
            Up(kernel_num >> 2, kernel_num >> 3, DoubleConv,
               normalization=normalization, leaky_relu=leaky_relu) for _ in range(input_c)
        ])
        self.decoder4 = nn.ModuleList([
            Up(kernel_num >> 3, kernel_num >> 4, DoubleConv,
               normalization=normalization, leaky_relu=leaky_relu) for _ in range(input_c)
        ])
        kernel_num >>= 4

        self.branch_out = nn.ModuleList([nn.Sequential(
            OrderedDict([
                ('branch_out_1', nn.Conv2d(kernel_num, output_c, kernel_size=3, stride=1, padding=1)),
                ('branch_out_2', obtain_normalization(choice=normalization, channels=output_c, groups=output_c)),
                ('branch_out_3', non_linear(inplace=True))
            ])
        ) for _ in range(input_c)])

        self.merge_out = OutConv(output_c * input_c, output_c)
        # self.bn1 = obtain_normalization(choice=normalization, channels=kernel_num >> 1, groups=output_c)
        # self.bn2 = obtain_normalization(choice=normalization, channels=kernel_num >> 2, groups=output_c)
        # self.bn3 = obtain_normalization(choice=normalization, channels=kernel_num >> 3, groups=output_c)
        # self.bn4 = obtain_normalization(choice=normalization, channels=kernel_num >> 4, groups=output_c)
        self.drop = nn.Dropout2d(p=p)
        self.bayesian = bayesian

    def _msfusion_2dpooling(self, tensor_tbp: torch.Tensor) -> torch.Tensor:
        pool_tensor = [self.pooling(tensor_tbp[i]) for i in range(tensor_tbp.shape[0])]
        pool_tensor = torch.stack(pool_tensor, dim=0)
        return pool_tensor

    def forward(self, x: torch.Tensor):
        # The input should follow the format of BCHW
        # print("shape:", x.shape)
        if x.shape[2] == 3:
            assert self.task == 'cls', "image channel should not be 3 when 'task' parameter is {}" \
                                       "with shape: {}".format(self.task, x.shape)
            x = x.transpose(0, 1)[0:2].unsqueeze(dim=2)
        input_device = 'cuda:%s' % x.get_device() if x.is_cuda else 'cpu'

        # x = torch.transpose(x, 0, 1).unsqueeze(2)
        # Branch_num x B x 1 x H x W
        x_inc = self.inc(x)  # Branch_numxBxCxHxW
        x_encoder1 = self.encoder1(self._msfusion_2dpooling(x_inc))
        x_encoder2 = self.encoder2(self._msfusion_2dpooling(x_encoder1))
        x_encoder3 = self.encoder3(self._msfusion_2dpooling(x_encoder2))
        x_encoder4 = self.encoder4(self._msfusion_2dpooling(x_encoder3))
        if self.bayesian:
            for i in range(x_encoder4.shape[0]):
                x_encoder4[i] = self.drop(x_encoder4[i])
        feat_decoded1 = torch.tensor([], device=input_device)
        feat_decoded2 = torch.tensor([], device=input_device)
        feat_decoded3 = torch.tensor([], device=input_device)
        feat_decoded4 = torch.tensor([], device=input_device)

        for i in range(x.shape[0]):
            feat_decoded1 = torch.cat([feat_decoded1, self.decoder1[i](x_encoder4[i], x_encoder3[i]).unsqueeze(0)],
                                      dim=0)
            feat_decoded2 = torch.cat([feat_decoded2, self.decoder2[i](feat_decoded1[-1], x_encoder2[i]).unsqueeze(0)],
                                      dim=0)
            feat_decoded3 = torch.cat([feat_decoded3, self.decoder3[i](feat_decoded2[-1], x_encoder1[i]).unsqueeze(0)],
                                      dim=0)
            feat_decoded4 = torch.cat([feat_decoded4, self.decoder4[i](feat_decoded3[-1], x_inc[i]).unsqueeze(0)],
                                      dim=0)

        logits = torch.tensor([], device=input_device)

        for i in range(x.shape[0]):
            branch_out = self.branch_out[i](self.drop(feat_decoded4[i]) if self.bayesian else feat_decoded4[i])
            logits = torch.cat([logits, branch_out], dim=1)
        logits = self.merge_out(logits)
        logits = logits.to(input_device)
        return logits
