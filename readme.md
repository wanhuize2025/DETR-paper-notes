# DETR 论文精读笔记

本项目是我在学习 **DETR (End-to-End Object Detection with Transformers)** 论文过程中整理的详细笔记，包含对论文核心思想、模型架构、损失函数、代码实现等多个方面的深入解析。

---

## 📁 目录结构

```text
.
├── code/                     # 相关代码（基于官方 DETR 实现）
│   └── detr/
├── figures/                  # 论文中的关键配图（用于笔记引用）
│   ├── Fig1.png
│   ├── Fig2.png
│   ├── ...
│   └── Fig16.png
├── notes/                    # 核心笔记（Markdown 格式）
│   ├── 自动驾驶DETR：DETR模型设计详解 .md
│   ├── 自动驾驶DETR：DETR目标检测和全景分割的损失函数是如何设计的.md
│   ├── 自动驾驶DETR：DETR目标检测和全景分割的检测头是如何设计的.md
│   ├── 自动驾驶DETR：DETR结果后处理和可视化详解.md
│   ├── 自动驾驶DETR：transformer编码器encoder结构是如何设计的.md
│   ├── 自动驾驶DETR：transformer解码器decoder结构是如何设计的.md
│   └── 自动驾驶DETR：输入transformer的特征图和位置编码是如何得到的.md
└── paper/                    # 原始论文（中英文版）
    ├── DETR-paper-CN.md
    └── DETR-paper-EN.pdf

```

## 📝 笔记内容概述

笔记共分为 7 个独立篇章，从不同角度剖析 DETR：

1. **[DETR 模型设计详解](./notes/自动驾驶DETR：DETR模型设计详解%20.md)**  
   整体介绍 DETR 的端到端目标检测流程，包括主干网络、Transformer 编码器-解码器、以及预测头的宏观设计。

2. **[Transformer 编码器（Encoder）结构设计](./notes/自动驾驶DETR：transformer编码器encoder结构是如何设计的.md)**  
   深入解析编码器的多头自注意力、前馈网络、残差连接和层归一化，以及如何利用自注意力学习全局特征。

3. **[Transformer 解码器（Decoder）结构设计](./notes/自动驾驶DETR：transformer解码器decoder结构是如何设计的.md)**  
   讲解解码器的交叉注意力机制、对象查询（Object Queries）的作用，以及解码器如何并行输出预测结果。

4. **[输入特征图和位置编码的获取方式](./notes/自动驾驶DETR：输入transformer的特征图和位置编码是如何得到的.md)**  
   详细说明 CNN 提取的二维特征如何展平并叠加位置编码（正弦/余弦编码），以保留空间信息。

5. **[检测头（Detection Head）设计](./notes/自动驾驶DETR：DETR目标检测和全景分割的检测头是如何设计的.md)**  
   介绍检测头的结构——FFN 预测框坐标和类别，以及全景分割头的设计思路。

6. **[损失函数设计](./notes/自动驾驶DETR：DETR目标检测和全景分割的损失函数是如何设计的.md)**  
   重点解析集合预测损失（基于二分匹配的 Hungarian Loss），包括分类损失和边界框损失（L1 + GIOU）的组合。

7. **[结果后处理与可视化](./notes/自动驾驶DETR：DETR结果后处理和可视化详解.md)**  
   探讨如何将模型输出转换为最终检测结果（如阈值过滤、NMS 与否）、以及预测结果的可视化方法。

## 🚀 如何使用

- 直接在 GitHub 上浏览 `.md` 文件，享受格式渲染。
- 若有本地阅读需求，推荐使用 vscode、Typora、Obsidian 等 Markdown 编辑器打开。
- 笔记中的图片均引用自 `figures/` 目录，确保路径正确即可显示。

## 📚 参考资料

- 原始论文：[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)  
- 官方代码：[Facebook Research DETR](https://github.com/facebookresearch/detr)

## 📌 说明

本仓库仅用于个人学习和知识分享，所有内容均为原创理解，如有错误欢迎指正或提交 Issue。

---

**Happy Learning!** 🎯
