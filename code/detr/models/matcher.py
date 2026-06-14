# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # 返回批量大小和查询数量（即模型预测的对象数量）。从模型的输出字典中获取"pred_logits"的形状，并提取批量大小和查询数量。
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # 将模型的输出进行展平，以便在批量中计算成本矩阵。
        # 具体来说，将"pred_logits"和"pred_boxes"的前两个维度（批量大小和查询数量）展平为一个维度，使得每个预测都成为一个独立的样本。
        # 这里的pred_logits通过softmax函数转换为概率分布，表示每个预测属于每个类别的概率。pred_boxes则保持原来的坐标形式。
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        # 将目标标签和边界框坐标进行连接，以便在计算成本矩阵时使用。
        # tgt_ids内容为每个目标的类别标签，tgt_bbox内容为每个目标的边界框坐标。
        # 通过将每个目标的标签和边界框坐标连接在一起，可以方便地计算每个预测与每个目标之间的分类成本和边界框成本。
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # 高级索引，根据目标类别标签从预测的概率分布中提取对应类别的概率值，并计算分类成本。
        # 具体来说，对于每个预测，使用目标类别标签作为索引，从预测的概率分布中提取对应类别的概率值，并计算分类成本为1减去该概率值。
        # 用1减去概率值是因为匈牙利匹配希望成本越低越好，而概率值越高表示预测越准确，因此成本应该随着概率值的增加而减少。
        cost_class = -out_prob[:, tgt_ids]
        #print("autodrv-cost_class:", cost_class.shape)

        # Compute the L1 cost between boxes
        # 计算边界框之间的L1成本，即预测边界框坐标与目标边界框坐标之间的绝对差值。
        # 计算公式为： cost_bbox = |predicted_box - target_box|，
        # 其中predicted_box是模型预测的边界框坐标，target_box是目标边界框坐标。
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        #print("autodrv-cost_bbox:", cost_bbox.shape)

        # Compute the giou cost betwen boxes
        # 计算边界框之间的GIoU成本，即预测边界框与目标边界框之间的广义交并比（GIoU）。
        # 计算公式为： cost_giou = -GIoU(predicted_box, target_box)，
        # 其中GIoU是预测边界框与目标边界框之间的广义交并比，取负值是因为匈牙利匹配
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        #print("autodrv-cost_giou:", cost_giou.shape)

        # Final cost matrix
        # 将分类成本、边界框成本和GIoU成本按照预设的权重进行加权求和，得到最终的成本矩阵C。
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # 将成本矩阵C的形状调整为[batch_size, num_queries, num_target_boxes]，以便后续使用线性求解器进行匹配。
        C = C.view(bs, num_queries, -1).cpu()
        #print("autodrv-C:", C.shape)

        # 获取每个批次中目标的数量
        sizes = [len(v["boxes"]) for v in targets]
        #print(f"autodrv-sizes: {sizes}, C.split(sizes, -1): {[c.shape for c in C.split(sizes, -1)]}")
        # 对于每个批次中的目标数量，使用线性求解器（linear_sum_assignment）在成本矩阵C中找到最佳的匹配关系。
        # 前方计算代价是多批次混合计算的，c[i]的索引则保证了不会跨批次进行匹配。线性求解器返回两个索引列表，分别表示选定的预测和对应的目标。
        # linear_sum_assignment函数在SciPy中实现，使用匈牙利算法来解决线性求解问题，找到成本矩阵C中每行和每列的最佳匹配关系，使得总成本最小。
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    # 构建匈牙利匹配器，传入分类成本、边界框成本和GIoU成本的权重参数。参数如下：
    # cost_class: 分类成本的权重，默认为1   
    # cost_bbox: 边界框成本的权重，默认为1
    # cost_giou: GIoU成本的权重，默认为1
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
