# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # 分类预测层：一个线性层，将Transformer解码器的输出映射到类别数加1（包括no-object类别）的维度上。
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # 边界框预测层：一个多层感知机（MLP），将Transformer解码器的输出映射到4维，
        # 表示边界框的中心坐标、宽度和高度。使用sigmoid函数将输出限制在[0, 1]范围内。
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # 使用嵌入层来生成查询向量，查询数为num_queries，维度为hidden_dim。
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # 输入投影层：一个卷积层，将backbone输出的特征图的通道数映射到hidden_dim，以便与Transformer的输入维度匹配。
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # 构建一个空类别权重张量，长度为类别数加1（包括no-object类别）。
        # 将no-object类别的权重设置为eos_coef，其他类别的权重设置为1。将该张量注册为模型的缓冲区，以便在训练过程中使用。
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        print(f"autodrv-SetCriterion-empty_weight:{empty_weight.shape}, {empty_weight}")
        # 注册一个名为empty_weight的缓冲区，保存类别权重张量。缓冲区是一种特殊的参数，不会被优化器更新，但会随模型一起保存和加载。
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        # 从模型的输出中获取分类预测的logits，并将其展平为二维张量，形状为[batch_size * num_queries, num_classes]。
        src_logits = outputs['pred_logits']

        #print("autodrv-indices:", indices)
        # 返回的批次索引和预测索引，用于从模型的输出中提取与目标匹配的预测。
        idx = self._get_src_permutation_idx(indices)
        #print("autodrv-idx:", idx)
        #print("autodrv-targets:", targets)
        # 按照匹配索引indices，从目标中提取对应的类别标签，并将其连接成一个一维张量，形状为[batch_size * num_matched_targets]。
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        #print("autodrv-target_classes_o:", target_classes_o)
        # 填充为no-object类别的标签，形状为[batch_size * num_queries]。对于匹配的预测，使用目标类别标签进行替换。
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # 将匹配的预测的标签替换为对应的目标类别标签。这里使用了索引idx来定位匹配的预测，并将其标签设置为target_classes_o中对应的值。
        # idx为一个元组，包含批次索引和预测索引。通过使用idx作为索引，可以将匹配的预测的标签替换为对应的目标类别标签，从而为后续的损失计算提供正确的标签信息。
        target_classes[idx] = target_classes_o
        #print("autodrv-target_classes:", target_classes)

        # 计算分类损失，使用交叉熵损失函数（cross_entropy）。
        # 将预测的logits转置为[batch_size * num_queries, num_classes]，并与目标类别标签进行比较。
        # 使用empty_weight作为权重，以便对no-object类别进行特殊处理。
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # 从预定义的损失映射中获取指定损失函数，并调用该函数计算损失值。损失映射是一个字典，键是损失名称，值是对应的损失函数。
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        # 调用对应的损失函数，传入模型的输出、目标、匹配索引、目标数量以及其他可能的参数，计算并返回损失值。
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # 在计算损失之前，首先从模型的输出中去除辅助输出（如果存在），以便只使用最后一层的输出进行匹配和损失计算。
        #print("outputs:", outputs)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # 计算模型的输出与目标之间的匹配关系，使用构建的matcher模块（通常是匈牙利匹配器）来计算最佳匹配关系。
        # matcher模块根据模型的输出和目标之间的成本矩阵，找到每个预测与哪个目标匹配。
        indices = self.matcher(outputs_without_aux, targets)
        print("autodrv-SetCriterion-HungarianMatcherindices:", indices)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # 求取总目标数量，并进行分布式同步，以便在多GPU训练中计算平均目标数量。这个值将用于后续的损失计算中的归一化处理。
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        print(f"autodrv-SetCriterion-self.losses:{self.losses}")
        for loss in self.losses:
            # 更新损失字典，调用get_loss方法计算每个指定的损失，并将其添加到损失字典中。
            # get_loss方法根据损失名称调用对应的损失函数来计算损失值。
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        print("autodrv-SetCriterion-computed losses:", losses)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        print(f"autodrv-PostProcess-out_logits:{out_logits.shape}, out_bbox:{out_bbox.shape}, target_sizes:{target_sizes.shape}")

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # 将预测的logits通过softmax函数转换为概率分布，表示每个预测属于每个类别的概率。
        prob = F.softmax(out_logits, -1)
        print(f"autodrv-PostProcess-softmax-prob:{prob.shape}, prob:{prob}")
        # 去掉最后一列的no-object类别概率，只保留前num_classes列的概率分布。
        # 然后，使用max函数找到每个预测的最高概率值和对应的类别标签（此处返回的是索引，正好对应类别标签）。
        scores, labels = prob[..., :-1].max(-1)
        print(f"autodrv-PostProcess-prob:{prob.shape}, scores:{scores.shape}, labels:{labels.shape}")

        # convert to [x0, y0, x1, y1] format
        # 将预测的边界框坐标从中心坐标格式（center_x, center_y, width, height）转换为角点坐标格式（x0, y0, x1, y1）。
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        # 将边界框坐标从相对坐标（相对于图像尺寸的归一化坐标）转换为绝对坐标（以像素为单位）。
        # 通过将边界框坐标乘以图像的宽度和高度，得到实际的像素坐标。
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # 获取每个预测的最高概率值（scores）、对应的类别标签（labels）和边界框坐标（boxes），
        # 并将它们组合成一个列表，每个元素是一个字典，包含'scores'、'labels'和'boxes'三个键，对应于每个预测的得分、类别标签和边界框坐标。
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    # 构建backbone
    backbone = build_backbone(args)
    # 构建transformer
    transformer = build_transformer(args)
    # 构建DETR模型，传入backbone、transformer、类别数、查询数和是否使用辅助损失等参数。参数如下：
    # backbone: 构建的骨干网络，用于提取图像特征
    # transformer: 构建的Transformer模块，用于处理特征图和查询向量
    # num_classes: 类别数，COCO数据集为91，其他数据集为20
    # num_queries: 查询数，即DETR能够检测的最大对象数量，COCO推荐使用100
    # aux_loss: 是否使用辅助损失，即在每个解码器层都计算损失，默认为False
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    # 构建匹配器，用于计算模型输出和目标之间的匹配关系。根据参数设置匹配器的类型和相关超参数。
    matcher = build_matcher(args)
    # 构建损失权重字典，指定不同损失的权重。根据参数设置分类损失、边界框损失、GIoU损失和掩码损失的权重。参数如下：
    # loss_ce: 分类损失的权重，默认为1
    # loss_bbox: 边界框损失的权重，默认为1
    # loss_giou: GIoU损失的权重，默认为1
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    # 构建损失计算器，传入类别数、匹配器、损失权重字典、no-object类别的权重和要计算的损失类型等参数。参数如下：
    # num_classes: 类别数，COCO数据集为91，其他数据集为20
    # matcher: 构建的匹配器，用于计算模型输出和目标之间的匹配关系
    # weight_dict: 损失权重字典，指定不同损失的权重
    # eos_coef: no-object类别的权重，默认为0.1
    # losses: 要计算的损失类型列表，包括'labels'（分类损失）、'boxes'（边界框损失）、'cardinality'（数量损失）
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    # 构建后处理器，用于将模型输出转换为COCO API期望的格式。根据参数设置后处理器的类型和相关超参数。参数如下：
    # postprocessors: 后处理器字典，包含'bbox'（边界框后处理器）等
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
