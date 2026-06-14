# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    # 初始化分布式训练环境
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # 设置设备cuda
    device = torch.device(args.device)

    # fix the seed for reproducibility
    # 在分布式训练中，每个进程的随机种子应该不同，以确保数据加载和模型训练的多样性。通过将基础种子
    # 与进程的rank相加，可以为每个进程生成一个独特的种子。
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 构建模型、损失函数和后处理器
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # 计算模型的可训练参数数量，并打印出来。这有助于了解模型的复杂度和训练资源需求。
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # 构建优化器，使用AdamW优化器，并将模型参数分为两组：
    # 一组是骨干网络的参数，使用较低的学习率，因为骨干网络的参数通常已经预训练好了；
    # 另一组是其他参数，使用默认的学习率。
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    # 构建学习率调度器，使用StepLR调度器，在指定的epoch数后降低学习率。
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # 构建训练和验证数据集，并根据是否使用分布式训练设置相应的数据采样器。对于训练数据集，使用随机采样器；对于验证数据集，使用顺序采样器。
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        # 在非分布式训练中，使用随机采样器对训练数据进行采样，以增加训练的多样性；对于验证数据，使用顺序采样器，以确保评估的一致性。
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 构建训练数据的批采样器，使用BatchSampler将随机采样器包装起来，以便在训练过程中按批次加载数据。
    # 设置drop_last=True以丢弃最后一个不完整的批次。
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    # 构建训练和验证数据加载器，使用DataLoader加载数据集，并传入相应的采样器、批采样器、数据增强变换函数和其他参数。
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        # 获取验证数据集的COCO API实例，以便在评估过程中使用。
        # 这个API提供了对COCO数据集的访问和操作功能，允许我们在评估阶段计算各种指标，如平均精度（AP）等。
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        # 从指定的检查点路径加载模型权重和训练状态，以便继续之前的训练过程。检查点可以是一个URL或本地文件路径。
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # 加载模型权重到当前模型中。这里使用model_without_ddp是因为在分布式训练中，
        # 模型被包装在DistributedDataParallel中，而model_without_ddp指向原始的模型实例。
        model_without_ddp.load_state_dict(checkpoint['model'])
        # 如果不是评估模式，并且检查点中包含优化器状态、学习率调度器状态和训练epoch信息，则加载这些状态以继续训练。
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        # 在评估模式下，直接调用evaluate函数对模型进行评估，并将评估结果保存到指定的输出目录中。评估结果包括测试统计信息和COCO评估器对象。
        # 参数如下：
        # model: 评估的模型实例。
        # criterion: 评估使用的损失函数实例。
        # postprocessors: 后处理器，用于处理模型的输出以适应评估的需求。
        # data_loader_val: 验证数据加载器，用于提供评估数据。
        # base_ds: 验证数据集的COCO API实例，用于计算评估指标。
        # device: 评估使用的设备（如CPU或GPU）。
        # args.output_dir: 评估结果保存的输出目录路径。
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        # 训练模型一个epoch，并获取训练统计信息。参数如下：
        # model: 训练的模型实例。
        # criterion: 训练使用的损失函数实例。
        # data_loader_train: 训练数据加载器，用于提供训练数据。
        # optimizer: 优化器实例，用于更新模型参数。
        # device: 训练使用的设备（如CPU或GPU）。
        # epoch: 当前训练的epoch数。
        # args.clip_max_norm: 梯度裁剪的最大范数，用于防止梯度爆炸。
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        # 更新学习率调度器，以便在训练过程中根据预设的策略调整学习率。
        lr_scheduler.step()
        if args.output_dir:
            # 保存当前训练状态的检查点，包括模型权重、优化器状态、学习率调度器状态、当前epoch数和训练参数等信息。
            # 检查点文件命名为'checkpoint.pth'，并在特定的epoch数（如每隔50个epoch）保存额外的检查点文件。
            # 每个epoch都要重新创建checkpoint_paths列表，以确保在每个epoch结束时都保存最新的检查点文件，
            # 并根据条件添加额外的检查点文件路径。
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            # 这里的逻辑是，在每个epoch结束时保存一个名为'checkpoint.pth'的检查点文件，
            # 并且在特定的epoch数（如每隔50个epoch或每隔100个epoch）保存额外的检查点文件，以便在训练过程中有更多的恢复点。
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                # 在每个epoch结束时，重新创建checkpoint_paths列表，并根据条件添加额外的检查点文件路径，
                # 以确保在每个epoch结束时都保存最新的检查点文件，并根据条件保存额外的检查点文件。
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                # 保存当前训练状态的检查点文件，包括模型权重、优化器状态、学习率调度器状态、当前epoch数和训练参数等信息。
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        # 在每个epoch结束后，评估模型在验证集上的性能，并获取测试统计信息和COCO评估器对象。
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )
        # 将训练统计信息和测试统计信息合并成一个日志字典，并添加当前epoch数和模型参数数量等信息，以便在训练过程中记录和分析模型的性能。
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        # 如果指定了输出目录，并且当前进程是主进程，则将日志统计信息保存到一个名为'log.txt'的文件中，
        # 并将COCO评估器的评估结果保存到指定的输出目录中。
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    # 计算总训练时间，并将其格式化为可读的字符串形式，然后打印出来，以便了解整个训练过程的耗时情况。
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
