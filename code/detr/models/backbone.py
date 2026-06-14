# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        # IntermediateLayerGetter是一个工具类，用于从一个模型中提取指定层的输出。
        # 它接受一个模型和一个字典，字典的键是要提取的层的名称，值是提取后的输出的名称。
        # 在这里，默认情况下，只有最后一层（layer4）的输出会被提取，但如果return_interm_layers为True，则会提取layer1、layer2、layer3和layer4的输出。
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        print(f"autodrv-BackboneBase: input tensor_list.tensors shape: {tensor_list.tensors.shape}, mask shape: {tensor_list.mask.shape if tensor_list.mask is not None else None}")
        xs = self.body(tensor_list.tensors)
        print(f"autodrv-BackboneBase: backbone output keys: {xs.keys()}, output shapes: {[x.shape for x in xs.values()]}")
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # 将输入的掩码（mask）调整为与输出特征图的空间尺寸相匹配。通过插值（interpolation）将掩码调整到输出特征图的大小，并转换为布尔类型。
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            print(f"autodrv-BackboneBase: output: {name}, shape: {x.shape}, mask shape: {mask.shape}")
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # 构建ResNet骨干网络，并使用FrozenBatchNorm2d替换标准的BatchNorm2d，以冻结批归一化层的参数。
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        # 调用父类的构造函数，传入构建的骨干网络、是否训练骨干网络、输出特征图的通道数以及是否返回中间层的特征图。
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        # 只用了Sequential存储方式
        # self.backbone = backbone
        # self.position_embedding = position_embedding
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        print(f"autodrv-Joiner: backbone output keys: {xs.keys()}, output shapes: {[x.tensors.shape for x in xs.values()]}")
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            #print(f"autodrv-Joiner: output: {name}, shape: {x.tensors.shape}")
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
            print(f"autodrv-Joiner: output: {name}, shape: {x.tensors.shape}, pos shape: {pos[-1].shape}")
        return out, pos


def build_backbone(args):
    # 构建位置编码器
    position_embedding = build_position_encoding(args)
    # 构建骨干网络（Backbone）。根据参数设置是否训练骨干网络，以及是否返回中间层的特征图。
    # 配置的骨干网络为resnet50
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # 将骨干网络和位置编码器组合成一个整体模型，并返回该模型。这个组合模型将同时输出骨干网络提取的特征图和对应的位置信息。
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
