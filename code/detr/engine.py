# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        # 计算加权总损失值，作为模型训练的目标函数，以便在训练过程中优化模型的性能。
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        # 反向传播计算梯度，并根据计算得到的梯度更新模型的参数，以便在训练过程中优化模型的性能。
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    # 将模型和损失函数设置为评估模式，以确保在评估过程中不进行梯度计算，并且模型的行为适合评估。
    model.eval()
    criterion.eval()

    # 创建一个MetricLogger实例，用于记录评估过程中的各种指标和统计信息。
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 添加分类错误率的度量器，以便在评估过程中记录这些指标的平均值。
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # 根据后处理器的类型，确定需要使用哪些评估器来评估模型的性能。
    # 这里检查后处理器中是否包含'segm'和'bbox'，并相应地创建CocoEvaluator实例。
    # 这里'segm'表示评估分割任务，'bbox'表示评估边界框检测任务。
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    # 遍历验证数据加载器中的每个批次数据，进行模型评估。
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # 将目标标签中的每个元素（如图像ID、边界框坐标等）移动到评估使用的设备上，以确保在评估过程中能够正确地计算损失和指标。
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 将输入样本传递给模型，获取模型的输出结果。
        outputs = model(samples)
        # 将模型的输出和对应的目标标签传递给损失函数，计算损失值和其他相关指标。
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # 将损失字典中的值进行缩放和未缩放的处理，以便在日志中记录这些指标的平均值。
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # 根据权重字典，对缩放后的损失值进行加权处理，以便在日志中记录这些指标的平均值。
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        # 将未缩放的损失值添加到日志中，以便在日志中记录这些指标的平均值。
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # 更新MetricLogger中的指标值，包括总损失、缩放后的损失值、未缩放的损失值以及分类错误率等，以便在评估过程中记录这些指标的平均值。
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # 获取图像的原始尺寸，并将其传递给后处理器，以便在评估过程中对模型的输出进行适当的处理和评估。
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # outputs输出的是归一化的边界框坐标和类别概率分布，orig_target_sizes包含了每个图像的原始尺寸信息。
        # 后处理器会根据这些信息将模型的输出转换为实际的边界框坐标和类别标签，以便进行评估。
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # 将评估结果与对应的图像ID进行匹配，一个图像中可能有多个预测结果。
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        #print(f"autodrv-res: {res}")
        if coco_evaluator is not None:
            # 把当前 batch 的预测结果加入 COCO evaluator，不断收集结果，最后在所有 batch 评估完成后进行汇总计算指标。
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    # 多卡同步
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        #汇总所有图片结果计算 precision和recall
        coco_evaluator.accumulate()
        # 输出AP、AP50、AP75、APS、APM、APL
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    # 生成最终stats字典，包含评估过程中记录的各种指标的平均值，以及COCO评估器和全景评估器的结果。
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    #print(f"autodrv-stats: {stats}, coco_evaluator: {coco_evaluator}")
    return stats, coco_evaluator
