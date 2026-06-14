# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        print(f"autodrv-PositionEmbeddingSine: input tensor_list.tensors shape: {x.shape}, mask shape: {mask.shape if mask is not None else None}")
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        #print(f"autodrv-PositionEmbeddingSine: y_embed: {y_embed}, x_embed: {x_embed}")
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 将位置编码分为两部分：x方向和y方向。对于每个位置，计算其在x和y方向上的位置编码。通过对位置进行除以dim_t来生成不同频率的正弦和余弦函数，从而为每个位置生成唯一的编码。
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        print(f"autodrv-PositionEmbeddingSine: pos_x: {pos_x.shape}, pos_y: {pos_y.shape}")
        # 对位置编码进行正弦和余弦变换。对于每个位置编码的维度，交替使用正弦和余弦函数来生成最终的位置编码。这样可以确保不同位置的编码具有唯一性，并且在训练过程中不需要学习参数。
        # .flatten(3)的作用是将最后一个维度（即位置编码的维度）展平，使得每个位置的编码成为一个一维向量。这样可以方便后续的处理和使用。
        # [sin1, cos1, sin2, cos2, sin3, cos3, ...]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        print(f"autodrv-PositionEmbeddingSine: pos_x after sin/cos: {pos_x.shape}, pos_y after sin/cos: {pos_y.shape}")
        # 将x和y方向的位置编码拼接在一起，并调整维度顺序。通过torch.cat将x和y方向的位置编码在最后一个维度上进行拼接，然后使用permute调整维度顺序，使得位置编码的维度在第二个位置，空间尺寸在后面。
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        print(f"autodrv-PositionEmbeddingSine: final pos shape: {pos.shape}")
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    # 位置编码的维度是隐藏层维度的一半，因为每个位置编码由正弦和余弦函数组成，每个函数占用一半的维度。
    N_steps = args.hidden_dim // 2
    print(f"autodrv-position encoding: {args.position_embedding}, with hidden_dim: {args.hidden_dim}, num_pos_feats: {N_steps}")
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        # 使用正弦位置编码（Sine Position Encoding）。
        # 这种编码方式基于正弦和余弦函数，能够为不同位置生成唯一的编码。它不需要学习参数，因此在训练过程中不会增加模型的复杂度。
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
