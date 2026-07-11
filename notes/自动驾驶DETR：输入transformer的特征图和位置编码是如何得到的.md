# 一文搞懂 DETR 中的特征图提取与位置编码细节

自从 Transformer 横扫 NLP 之后，视觉领域的 **DETR**（Detection Transformer）也成了目标检测的热门话题。很多熟悉 CNN 和 YOLO、Faster R-CNN 的朋友，第一次看到 DETR 时都会困惑：**特征图怎么喂给 Transformer？位置编码又从哪来？** 今天我们就结合 DETR 源码，把这两个核心问题讲透。

---

## 一、特征图和位置编码在 DETR 中的"生态位"

先上一张简化的 DETR 流程图：

```
输入图像 → Backbone（CNN）→ 特征图 + 位置编码 → Transformer Encoder → ... → 预测
```

在 DETR 中，**Transformer Encoder** 的输入不是单纯的图像 patch 序列，而是两样东西：

1. **特征图**：由 CNN backbone（如 ResNet50）从原图提取的 **高维特征图**，形状为 `(C, H, W)`。它相当于把图像编码成一组视觉"单词"。
2. **位置编码**：与特征图 **相同空间尺寸** 的位置嵌入，用于告诉 Transformer 每个像素点在原图中的相对或绝对位置。

这两者在进入 Encoder 之前会 **逐元素相加**（不是拼接）。简单说：特征图是"**画布上的颜色**"，位置编码是"**每个像素的坐标标签**"。

---

## 二、特征图是怎么生成的？——Backbone 的"烹饪过程"

### 2.1 整体装配：`build_backbone` 函数

我们首先看 `build_backbone` 这个"总装车间"：

```python
def build_backbone(args):
    # 构建位置编码器
    position_embedding = build_position_encoding(args)
    
    # 构建骨干网络（Backbone）。根据参数设置是否训练骨干网络，以及是否返回中间层的特征图。
    # 配置的骨干网络为 resnet50
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    
    # 将骨干网络和位置编码器组合成一个整体模型，并返回该模型。
    # 这个组合模型将同时输出骨干网络提取的特征图和对应的位置信息。
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
```

这里的关键是 `Joiner` 类——它把 **backbone** 和 **位置编码器** 组合起来。当输入图像经过 `Joiner` 时，会同时输出：
- **特征图**（从 backbone 得到）
- **位置编码**（从位置编码器得到，基于特征图的尺寸）

### 2.2 核心食材：`Backbone` 类

`Backbone` 类继承自 `BackboneBase`，代码中默认使用 **ResNet50**：

```python
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # 构建 ResNet 骨干网络，并使用 FrozenBatchNorm2d 替换标准的 BatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), 
            norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        # 调用父类的构造函数
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
```

**关键参数解析**：
- **`replace_stride_with_dilation=[False, False, dilation]`**：控制 ResNet 各 stage 是否用空洞卷积替换 stride
- **`FrozenBatchNorm2d`**：冻结 BN 层，稳定训练
- **`num_channels`**：ResNet50/101 输出 **2048 维**，这就是特征图的通道数

父类 `BackboneBase` 的核心逻辑：

```python
class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, 
                 num_channels: int, return_interm_layers: bool):
        super().__init__()
        # 冻结不需要训练的层
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        # 决定返回哪些层的特征图
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
    
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # 将 mask 插值到特征图相同尺寸
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
```

默认情况下，`return_interm_layers=False`，只会返回 **最后一个 stage（layer4）的特征图**，形状为 `(2048, H/32, W/32)`。

> 🧪 **比喻**：ResNet 像是一个多层压榨机，每一层都在提炼图像语义，最后榨出 2048 张浓缩的"语义地图"。

### 2.3 为什么 DETR 离不开这个 Backbone？

有人可能会想：为什么不直接把原始图像像素送进 Transformer？原因有三：

1. **计算量爆炸**：一张 800×1066 的图像就有 85 万个像素点，Self-Attention 复杂度 $O(n^2)$，完全不可行。
2. **语义信息不足**：像素级信息太底层，缺少物体级别的上下文。
3. **DETR 论文实验证明**：移除 backbone 性能会断崖式下跌。

> **比喻**：Backbone 就像一台"智能相机"，先对图像进行"粗加工"——提取边缘、纹理、形状，最终输出一张"语义地图"，Transformer 在这张地图上做检测。

### 2.4 ⭐ 重点：dilation 是什么？作用是什么？在哪里引入？

**空洞卷积（Dilated Convolution）**：通过在卷积核元素之间插入"空洞"，让卷积核在不增加参数量的情况下拥有更大的感受野。

**在 DETR 中的作用**：
- 默认 ResNet 的 layer4 有 stride=2，输出步长为 32。开启 `dilation=True` 后，layer4 的 stride=2 被替换为 dilation=2 的空洞卷积，**输出步长保持为 16**，特征图分辨率翻倍。
- **为什么需要更高分辨率？** 小物体检测！步长 16 时小物体信息保留更多。

**在哪里引入**：在 `Backbone.__init__` 中，通过 `replace_stride_with_dilation=[False, False, dilation]` 参数传给 ResNet 的构造器。

```python
# 当 dilation=True 时
replace_stride_with_dilation=[False, False, True]
# 表示：layer2 无变化，layer3 无变化，layer4 将 stride=2 替换为 dilation=2
```

### 2.5 特征图进入 Transformer 前的"最后一步"

当 `Joiner` 执行 forward 时：

```python
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
    
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)   # 先过 backbone
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos
```

特征图会拉直成一个序列，形状从 `(C, H, W)` 变为 `(H \times W, C)`，再送入 Transformer。

### 2.6 动手跑一跑 Backbone：从输入到输出的真实变换

> 理论说了这么多，不如直接看代码跑出来的结果。我们通过打印日志，完整追踪 Backbone 如何处理一张图像。

需要注意的是，DETR 为了自适应不同尺寸的图片，使用了 `NestedTensor` 结构，其中包含了 **填充（padding）** 和 **掩码（mask）** 技术。

下面我们分别测试 `dilation=False` 和 `dilation=True` 两种情况：

#### 🔹 当 `dilation=False`（默认配置，步长=32）

```
autodrv-BackboneBase: input tensor_list.tensors shape: torch.Size([2, 3, 800, 1332]), mask shape: torch.Size([2, 800, 1332])
autodrv-BackboneBase: backbone output keys: odict_keys(['0']), output shapes: [torch.Size([2, 2048, 25, 42])]
autodrv-BackboneBase: output: 0, shape: torch.Size([2, 2048, 25, 42]), mask shape: torch.Size([2, 25, 42])
```

**解读**：
- 输入：2 张 800×1332 的 RGB 图像
- 输出：$2 \times 2048 \times 25 \times 42$ 的特征图
- 空间下采样倍数：$800 \div 25 = 32$，$1332 \div 42 \approx 31.7$
- **步长=32**：每个特征点对应原图 $32 \times 32$ 的区域

#### 🔸 当 `dilation=True`（启用空洞卷积，步长=16）

```
autodrv-BackboneBase: input tensor_list.tensors shape: torch.Size([2, 3, 800, 1332]), mask shape: torch.Size([2, 800, 1332])
autodrv-BackboneBase: backbone output keys: odict_keys(['0']), output shapes: [torch.Size([2, 2048, 50, 84])]
autodrv-BackboneBase: output: 0, shape: torch.Size([2, 2048, 50, 84]), mask shape: torch.Size([2, 50, 84])
```

**解读**：
- 输出：$2 \times 2048 \times 50 \times 84$ 的特征图
- 空间下采样倍数：$800 \div 50 = 16$，$1332 \div 84 \approx 15.8$
- **步长=16**：特征图分辨率翻倍，每个特征点对应原图 $16 \times 16$ 的区域

> ✅ **结论**：`dilation=True` 使特征图分辨率从 $25 \times 42$ 提升到 $50 \times 84$，**像素点数量增加 4 倍**，尤其有利于小物体检测。同时，mask 也会被同步插值到特征图尺寸，保证有效区域标记正确。

---

## 三、位置编码是怎么产生的？——给每个像素发"身份证"

### 3.1 位置编码的构建入口

DETR 支持两种位置编码：**正弦（sine）** 和 **可学习（learned）**，默认使用正弦。构建代码：

```python
def build_position_encoding(args):
    # 位置编码的维度是隐藏层维度的一半
    N_steps = args.hidden_dim // 2   # hidden_dim=256 → N_steps=128
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")
    return position_embedding
```

为什么 `N_steps = hidden_dim // 2`？因为对于每个位置，正弦和余弦函数各负责一半维度。

### 3.2 ⭐ 二维位置编码的详细实现

图像是二维的，因此我们需要分别对 **x 坐标** 和 **y 坐标** 进行编码，然后将两个方向的编码拼接起来。

**核心公式**（和 Transformer 原文一致）：

对于特征图上的一个点 $(x, y)$，其位置编码的第 $i$ 维定义为：

**对 y 坐标编码：**

$$
\begin{aligned}
PE_{(y, 2i)} &= \sin\left(\frac{y}{10000^{2i / d}}\right) \\
PE_{(y, 2i+1)} &= \cos\left(\frac{y}{10000^{2i / d}}\right)
\end{aligned}
$$

**对 x 坐标编码：**

$$
\begin{aligned}
PE_{(x, 2i)} &= \sin\left(\frac{x}{10000^{2i / d}}\right) \\
PE_{(x, 2i+1)} &= \cos\left(\frac{x}{10000^{2i / d}}\right)
\end{aligned}
$$

其中 $d = N\_steps = hidden\_dim // 2$，$i$ 从 0 到 $d-1$。

最终位置编码 = $[PE_y, PE_x]$，总维度 $2d = hidden\_dim$。

**代码中的归一化细节**：

```python
# PositionEmbeddingSine 的核心逻辑（简化）
def forward(self, x):
    not_mask = torch.ones_like(x[0, 0])   # (H, W)
    y_embed = not_mask.cumsum(0, dtype=torch.float32)  # y 坐标从 0 到 H-1
    x_embed = not_mask.cumsum(1, dtype=torch.float32)  # x 坐标从 0 到 W-1
    
    if self.normalize:
        eps = 1e-6
        # 归一化到 [0, 2π]
        y_embed = y_embed / (y_embed[-1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, -1:] + eps) * 2 * math.pi
    
    # 对 x_embed 和 y_embed 分别应用 sin/cos，并拼接
    return pos_encoding   # shape: (C, H, W)
```

> **比喻**：正弦位置编码就像一个"经纬度网格"——每个点都有唯一的经纬度坐标，并且相邻点的坐标连续变化。

### 3.3 可学习位置编码（备选）

如果选择 `learned` 版本，则直接初始化一个可训练的参数矩阵，形状为 $(H, W, C)$，随网络一起优化。但 DETR 论文实验表明正弦编码效果略好且无需额外参数。

### 3.4 动手跑一跑位置编码：从坐标网格到正弦编码

> 位置编码是如何从一张"空白坐标图"变成 256 维的"位置指纹"？我们打印每一步的形状变化。

输入是 Backbone 输出的特征图（以 `dilation=False` 为例，尺寸 $2 \times 2048 \times 25 \times 42$）：

```
autodrv-PositionEmbeddingSine: input tensor_list.tensors shape: torch.Size([2, 2048, 25, 42]), mask shape: torch.Size([2, 25, 42])
```

**第一步：生成 x 和 y 坐标网格**

```
autodrv-PositionEmbeddingSine: pos_x: torch.Size([2, 25, 42, 128]), pos_y: torch.Size([2, 25, 42, 128])
```

**第二步：应用 sin/cos 交替编码**

```
autodrv-PositionEmbeddingSine: pos_x after sin/cos: torch.Size([2, 25, 42, 128]), pos_y after sin/cos: torch.Size([2, 25, 42, 128])
```

**第三步：拼接 y 和 x 编码**

```
autodrv-PositionEmbeddingSine: final pos shape: torch.Size([2, 256, 25, 42])
```

**最终结果**：位置编码的形状为 $(batch=2, channels=256, H=25, W=42)$，与降维后的特征图完全匹配，注意这里做了一次通道permute，将(B,H,W,C)改成了PyTorch CNN 特征图常用格式： (B,C,H,W)。

> 📌 **关键点**：前 128 维编码的是 **y 坐标**（垂直方向），后 128 维编码的是 **x 坐标**（水平方向）。

---

## 四、总结：两者如何配合舞动？

| 组件 | 来源 | 作用 | 形状 |
|------|------|------|------|
| **特征图** | ResNet backbone 最后一个 stage | 提供 **语义内容**：这是什么东西？ | $(2048, H/32, W/32)$ 或 $(2048, H/16, W/16)$（dilation=True） |
| **位置编码** | 正弦函数动态生成 | 提供 **空间位置**：这个东西在图像哪里？ | $(256, H/32, W/32)$ 或 $(256, H/16, W/16)$ |

**合并方式**：
1. 特征图先通过一个 `nn.Conv2d(2048, 256, 1)` 降维到与位置编码相同的通道数 256
2. 然后逐元素相加：$encoder\_input = feature\_map\_proj + position\_encoding$
3. 拉平成序列 $(H \times W, batch, 256)$ 送入 Transformer

> 💡 **直观比喻**：特征图是 **电影的画面内容**（人物、背景），位置编码是 **每个像素的经纬度**。没有位置编码，Transformer 会以为所有像素都堆在同一个"混沌点"上。

**完整前向流程回顾**：

```python
# 假设输入图像 x，尺寸 (batch, 3, H, W)
# 1. Backbone 提取特征图
features = backbone(x)  # Dict: {'0': NestedTensor(tensors=(batch,2048,h,w), mask=...)}
# 2. 取特征图的 tensors 部分
feature_map = features['0'].tensors  # (batch, 2048, h, w)
# 3. 降维到 hidden_dim
feature_map_proj = conv1x1(feature_map)  # (batch, 256, h, w)
# 4. 生成位置编码
pos_encoding = position_embedding(features['0'])  # (batch, 256, h, w)
# 5. 相加并拉平
encoder_input = (feature_map_proj + pos_encoding).flatten(2).permute(2,0,1)  # (h*w, batch, 256)
```

现在你知道了：**Backbone 负责煮饭（特征图），位置编码负责贴标签（坐标信息）**。两者结合，才让 Transformer 看懂图像。希望这篇文章能帮你跨过从 CNN 到 Transformer 的第一道门槛！