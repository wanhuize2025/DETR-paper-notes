# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        # 调用父类的构造函数，初始化COCO数据集，并传入图像文件夹路径和标注文件路径。
        super(CocoDetection, self).__init__(img_folder, ann_file)
        # 将数据增强变换和是否返回掩码的参数保存为实例变量，以便在后续的数据处理过程中使用。
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        # 从数据集中获取图像和对应的标注信息。首先调用父类的__getitem__方法获取原始的图像和标注数据。
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        # 将图像ID和标注信息封装成一个字典，并传递给数据增强变换函数进行处理。最后返回处理后的图像和标注数据。
        target = {'image_id': image_id, 'annotations': target}
        #print(f"image_id: {image_id}, annotations: {target['annotations']} \n")
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            # 增强处理
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        # 只考虑是单个目标的情况，对于一群难以区分的物体集合不考虑
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        # 提取图像中每个目标的边界框坐标
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # COCO数据集中，边界框坐标是以左上角的坐标和宽高的形式给出的，因此需要将宽高转换为右下角的坐标。
        # 具体来说，将边界框的宽度加到左上角的x坐标上，将边界框的高度加到左上角的y坐标上，从而得到右下角的坐标。
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # 提取图像中每个目标的类别标签，并将其转换为PyTorch张量。
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # 提取图像中每个目标的关键点信息（如果存在）。如果标注中包含关键点信息，则将其转换为PyTorch张量，并调整其形状以适应后续的处理。
        # 这里没有这个信息，keypoints为None
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)
        # 过滤掉无效的边界框，即那些宽度或高度为零的边界框。
        # 具体来说，检查每个边界框的右下角坐标是否大于左上角坐标，如果不满足条件，则认为该边界框无效并将其过滤掉。
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]


        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # 将处理后的边界框坐标、类别标签、掩码（如果返回）和关键点信息（如果存在）封装成一个字典，并返回处理后的图像和该字典。
        # 这个字典包含了图像中每个目标的相关信息，供后续的模型训练或评估使用。
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        # 过滤掉无效的边界框对应的面积和iscrowd信息，以保持与过滤后的边界框和类别标签的一致性。
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # 将图像的原始尺寸和处理后的尺寸封装成一个字典，并返回处理后的图像和该字典。
        # 这个字典包含了图像的原始尺寸和处理后的尺寸，供后续的模型训练或评估使用。
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    # 构建COCO数据集，使用CocoDetection类，并传入图像文件夹路径、标注文件路径、
    # 数据增强变换以及是否返回掩码的参数。数据增强变换根据训练或验证集的不同而有所区别。
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
