# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


class DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        # 注意力模块，输入是transformer的输出和backbone的输出，输出是一个注意力权重图，参数如下：
        # hidden_dim是transformer输出的特征维度，
        # nheads是注意力头的数量，
        # dropout是注意力权重的dropout概率，
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        # 卷积头，输入是transformer的输出和注意力权重图，以及backbone的中间特征图，输出是一个分割掩码
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

        print(f"autodrv-DETRsegm-self.detr:{self.detr}, self.bbox_attention:{self.bbox_attention}, self.mask_head:{self.mask_head}")

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # 从DETR模型的backbone中获取特征图和位置编码。backbone的输入是一个NestedTensor，输出是一个包含特征图和位置编码的元组。
        features, pos = self.detr.backbone(samples)

        # 获取批量大小
        bs = features[-1].tensors.shape[0]

        # 从DETR模型的transformer中获取预测的类别和边界框坐标。
        src, mask = features[-1].decompose()
        assert mask is not None
        # 通过1x1卷积将transformer的输出特征图的通道数调整为hidden_dim，以便后续的注意力计算和卷积头处理。
        src_proj = self.detr.input_proj(src)
        # 通过transformer计算得到hs和memory，其中hs是transformer的输出，memory是transformer的记忆特征图。
        # 输入包括调整后的特征图src_proj、掩码mask、查询嵌入query_embed.weight和位置编码pos[-1]。
        hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])

        # 通过线性层class_embed和bbox_embed分别预测类别和边界框坐标。class_embed的输入是hs，输出是预测的类别logits；
        # bbox_embed的输入也是hs，输出是预测的边界框坐标，并通过sigmoid函数将坐标归一化到[0,1]范围内。
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.detr.aux_loss:
            out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)

        # FIXME h_boxes takes the last one computed, keep this in mind
        # 通过注意力模块计算得到bbox_mask，输入是transformer的输出hs[-1]和记忆特征图memory，以及掩码mask。
        # bbox_mask是一个注意力权重图，表示每个查询与特征图中每个位置的相关性。参数如下：
        # hs[-1]是transformer的最后一层输出，形状为[batch_size, num_queries, hidden_dim]，
        # memory是transformer的记忆特征图，形状为[batch_size, hidden_dim, height, width]，
        # mask是输入图像的掩码，形状为[batch_size, height, width]，其中True表示被掩盖的位置，False表示有效的位置。
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
        # 通过卷积头计算得到seg_masks，输入是transformer的输出hs[-1]和注意力权重图bbox_mask，
        # 以及backbone的中间特征图features[2]、features[1]和features[0]。参数如下：
        # src_proj是backbone输出的特征图，形状为[batch_size, hidden_dim, height, width]，
        # bbox_mask是注意力权重图，形状为[batch_size, num_queries, height, width]，
        # features[2].tensors、features[1].tensors和features[0].tensors是backbone的中间特征图，形状为[batch_size, hidden_dim, height, width]。
        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        # 将seg_masks的形状调整为[batch_size, num_queries, height, width]，其中num_queries是transformer的输出查询数量，
        # height和width是分割掩码的空间尺寸。
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

        out["pred_masks"] = outputs_seg_masks
        return out


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        # context_dim值为256
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        print(f"autodrv-MaskHeadSmallConv-inter_dims:{inter_dims}")
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        #  将输入的特征图x和注意力权重图bbox_mask进行拼接，作为卷积头的输入。具体来说，将bbox_mask的形状调整为与x相同，并在通道维度上进行拼接。
        print(f"autodrv-MaskHeadSmallConv-x:{x.shape}, bbox_mask:{bbox_mask.shape}, fpns:{[f.shape for f in fpns]}")
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        # 2层卷积，每层卷积后进行group norm和ReLU激活函数处理。卷积层的输入和输出通道数由inter_dims列表定义，具体如下：
        # 第一层卷积的输入通道数为dim，输出通道数为dim，卷积核大小为3，填充为1；
        # 第二层卷积的输入通道数为dim，输出通道数为inter_dims[1]，卷积核大小为3，填充为1；
        print(f"autodrv-MaskHeadSmallConv-x.shape-1:{x.shape}")
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)
        print(f"autodrv-MaskHeadSmallConv-x.shape-2:{x.shape}")

        cur_fpn = self.adapter1(fpns[0])
        print(f"autodrv-MaskHeadSmallConv-cur_fpn.shape-1:{cur_fpn.shape},x.size(0):{x.size(0)}, cur_fpn.size(0):{cur_fpn.size(0)}, fpns[0]:{fpns[0].shape}, cur_fpn:{cur_fpn.shape}")
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)
        print(f"autodrv-MaskHeadSmallConv-x.shape-3:{x.shape}")

        cur_fpn = self.adapter2(fpns[1])
        print(f"autodrv-MaskHeadSmallConv-cur_fpn.shape-2:{cur_fpn.shape},x.size(0):{x.size(0)}, cur_fpn.size(0):{cur_fpn.size(0)}")
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)
        print(f"autodrv-MaskHeadSmallConv-x.shape-4:{x.shape}")

        cur_fpn = self.adapter3(fpns[2])
        print(f"autodrv-MaskHeadSmallConv-cur_fpn.shape-3:{cur_fpn.shape},x.size(0):{x.size(0)}, cur_fpn.size(0):{cur_fpn.size(0)}")
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        print(f"autodrv-MaskHeadSmallConv-x.shape-4-1:{x.shape}")
        x = self.lay5(x)
        print(f"autodrv-MaskHeadSmallConv-x.shape-4-2:{x.shape}")
        x = self.gn5(x)
        x = F.relu(x)
        print(f"autodrv-MaskHeadSmallConv-x.shape-5:{x.shape}")

        x = self.out_lay(x)
        print(f"autodrv-MaskHeadSmallConv-x.shape-6:{x.shape}")
        return x

# 缩放点积注意力，只计算注意力权重，不进行值的乘法；
# 这个函数相当于transformer解码器的输出与backbone的输出之间的交叉注意力机制，
# 输入是transformer的输出和backbone的输出，以及掩码，输出是一个注意力权重图。
class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""


    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # 定义线性层q_linear和k_linear，用于将输入的查询和键映射到隐藏维度空间。query_dim是输入查询和键的维度，
        # hidden_dim是输出的维度，num_heads是注意力头的数量。
        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        # 初始化线性层的权重和偏置。对于k_linear和q_linear，权重使用Xavier均匀分布进行初始化，偏置初始化为零。
        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        # 输入q和k分别是查询和键的张量，形状为[batch_size, num_queries, query_dim]和[batch_size, num_keys, query_dim]。
        q = self.q_linear(q)
        # 相当于对k进行线性变换，将其从输入维度映射到隐藏维度。
        # k_linear.weight的形状为[hidden_dim, query_dim]，通过unsqueeze将其调整为[hidden_dim, query_dim, 1, 1]，以便进行卷积操作。
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        # 将q和k的形状调整为适合多头注意力计算的形式。q被调整为[batch_size, num_queries, num_heads, hidden_dim // num_heads]，
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        # 计算注意力权重。通过爱因斯坦求和（einsum）将q和k进行矩阵乘法，得到一个形状为[batch_size, num_queries, num_heads, height, width]的张量weights，
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            # 如果提供了掩码mask，则将权重中被掩盖的位置设置为负无穷，以便在softmax计算中被忽略。mask的形状为[batch_size, height, width]，其中True表示被掩盖的位置，False表示有效的位置。
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        # 通过softmax函数将权重转换为概率分布，表示每个查询与特征图中每个位置的相关性。
        # 首先将权重展平为[batch_size, num_queries, num_heads, height * width]，
        # 然后在最后一个维度上应用softmax，最后将权重的形状调整回原来的形式。
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        print(f"autodrv-PostProcessSegm-orig_target_sizes:{orig_target_sizes}, max_target_sizes:{max_target_sizes}")
        print(f"autodrv-PostProcessSegm-outputs['pred_logits']:{outputs['pred_logits'].shape}, outputs['pred_boxes']:{outputs['pred_boxes'].shape}, outputs['pred_masks']:{outputs['pred_masks'].shape}")
        assert len(orig_target_sizes) == len(max_target_sizes)
        # 计算所有图像中最大的高度和宽度，以便后续对分割掩码进行插值处理。
        # max_target_sizes是一个形状为[batch_size, 2]的张量，其中每行包含一个图像的最大高度和宽度。
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        print(f"autodrv-PostProcessSegm-outputs_masks:{outputs_masks.shape}, max_h:{max_h}, max_w:{max_w}")
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        print(f"autodrv-PostProcessSegm-outputs_masks-interpolate:{outputs_masks.shape}, max_h:{max_h}, max_w:{max_w}")
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()
        print(f"autodrv-PostProcessSegm-outputs_masks-sigmoid-threshold:{outputs_masks.shape}, max_h:{max_h}, max_w:{max_w}")

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            print(f"autodrv-PostProcessSegm-results[i]['masks']-1:{results[i]['masks'].shape}, img_h:{img_h}, img_w:{img_w}")
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()
            print(f"autodrv-PostProcessSegm-results[i]['masks']-2:{results[i]['masks'].shape}, tt:{tt.tolist()}")

        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())
        print(f"autodrv-PostProcessPanoptic-out_logits:{out_logits.shape}, raw_masks:{raw_masks.shape}, raw_boxes:{raw_boxes.shape}, processed_sizes:{processed_sizes}, target_sizes:{target_sizes}")
        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            print(f"autodrv-PostProcessPanoptic-cur_logits:{cur_logits.shape}, cur_masks:{cur_masks.shape}, cur_boxes:{cur_boxes.shape}, size:{size}, target_size:{target_size}")
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])
            print(f"autodrv-PostProcessPanoptic-cur_scores:{cur_scores.shape}, cur_classes:{cur_classes.shape}, cur_masks:{cur_masks.shape}, cur_boxes:{cur_boxes.shape}")


            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image
                print(f"autodrv-PostProcessPanoptic-get_ids_area-masks:{masks.shape}, scores:{scores.shape}, dedup:{dedup}")
                m_id = masks.transpose(0, 1).softmax(-1)
                print(f"autodrv-PostProcessPanoptic-get_ids_area-m_id:{m_id.shape}")

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)
                print(f"autodrv-PostProcessPanoptic-get_ids_area-m_id-argmax:{m_id.shape}, {m_id}")

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds
