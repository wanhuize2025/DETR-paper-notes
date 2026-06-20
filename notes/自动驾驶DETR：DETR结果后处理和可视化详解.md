# DETR结果后处理和可视化详解

> 从模型输出到COCO格式，一步步拆解DETR的后处理与可视化流程

## 官方资源

在深入技术细节之前，先给大家推荐几个官方的Colab Notebook，可以快速上手体验DETR：

- **[DETR's hands on Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb)**：展示如何从hub加载模型、生成预测，并可视化模型的注意力机制（与论文中的示意图类似）

- **[Standalone Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb)**：用50行Python代码从零实现简化版DETR并可视化预测结果。如果你想深入理解架构，这是最佳的起点

- **[Panoptic Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb)**：演示如何使用DETR进行全景分割并绘制预测结果

这些Notebook是理解DETR的绝佳资源，里面有详细的可视化效果展示，建议先跑一遍再来看本文，会有更深的体会。

## 前言

DETR（Detection Transformer）作为目标检测领域的革新之作，其端到端的设计理念让人眼前一亮。但真正在生产环境中使用DETR时，**后处理**环节往往是决定最终效果的关键一步。

本文将深入剖析DETR的三种后处理方式：`postprocessors['bbox']`、`postprocessors['segm']`和`postprocessors["panoptic"]`，并结合实际运行日志详细解析Panoptic后处理的完整流程。最后，我们还会介绍如何将后处理结果进行可视化。

## 一、三种后处理器概览

在DETR的`build`函数中，我们可以看到后处理器的构建逻辑：

```python
postprocessors = {'bbox': PostProcess()}
if args.masks:
    postprocessors['segm'] = PostProcessSegm()
if args.dataset_file == "coco_panoptic":
    is_thing_map = {i: i <= 90 for i in range(201)}
    postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)
```

三种后处理器各司其职：

| 后处理器 | 用途 | 输出格式 |
|---------|------|---------|
| `PostProcess` (bbox) | 目标检测 | 边界框 + 类别 + 置信度 |
| `PostProcessSegm` (segm) | 实例分割 | 边界框 + 类别 + 置信度 + 掩码 |
| `PostProcessPanoptic` (panoptic) | 全景分割 | PNG图像 + segments_info |

## 二、PostProcess（边界框后处理）

这是最基础的后处理器，将模型输出转换为COCO检测格式：

```python
class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        # Step 1: Softmax得到概率分布
        prob = F.softmax(out_logits, -1)
        
        # Step 2: 去掉no-object类别，取最大概率
        scores, labels = prob[..., :-1].max(-1)
        
        # Step 3: 边界框坐标转换 (cx, cy, w, h) -> (x0, y0, x1, y1)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        
        # Step 4: 从归一化坐标转为绝对像素坐标
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        
        return [{'scores': s, 'labels': l, 'boxes': b} 
                for s, l, b in zip(scores, labels, boxes)]
```

**关键点**：
- 使用`softmax`而非`sigmoid`，因为DETR的类别预测是互斥的
- 自动过滤掉`num_classes`（no-object类别）
- 坐标从相对值（0~1）转换为绝对值（像素）

## 三、PostProcessSegm（实例分割后处理）

在边界框基础上增加了掩码预测：

```python
class PostProcessSegm(nn.Module):
    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        # Step 1: 插值到最大尺寸
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = F.interpolate(
            outputs_masks, 
            size=(max_h, max_w), 
            mode="bilinear", 
            align_corners=False
        )
        
        # Step 2: Sigmoid + 阈值二值化
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()
        
        # Step 3: 裁剪到实际图像尺寸
        for i, (cur_mask, t, tt) in enumerate(zip(...)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            
            # Step 4: 缩放到原始尺寸（使用NEAREST保持离散性）
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), 
                size=tuple(tt.tolist()),
                mode="nearest"
            ).byte()
```

**关键点**：
- 先上采样到batch内最大尺寸，再裁剪到实际尺寸
- 使用`nearest`插值而非双线性，保证掩码是二值的
- 阈值默认为0.5

## 四、PostProcessPanoptic（全景分割后处理）⭐

这是本文的重点，也是DETR中最复杂的后处理器。让我们拆解成8个阶段：

### 输入数据

```
cur_masks.shape = [N, H*W]      # N个query的mask logits
cur_classes.shape = [N]          # 每个query的类别
cur_scores.shape = [N]           # 每个query的置信度
```

### 第一阶段：生成像素级实例ID图

```python
# Step 1: 转置 + Softmax
m_id = masks.transpose(0, 1).softmax(-1)
# 形状: [H*W, N] -> 每个像素对应N个mask的分数

# Step 2: Argmax得到每个像素归属的mask索引
m_id = m_id.argmax(-1).view(h, w)
# 形状: [h, w] -> 每个像素的值表示属于哪个mask
```

**举例说明**：
```
假设masks.shape = [3, 6]，即3个mask，6个像素

转置后每个像素对应3个分数：
像素0: [2.1, 0.3, -1.5] -> softmax -> [0.85, 0.12, 0.03] -> argmax -> 0
像素1: [0.2, 3.8, -0.4] -> softmax -> [0.02, 0.95, 0.03] -> argmax -> 1
...

最终m_id = [[0, 1, 2], [0, 1, 2]]  # 2x3
```

### 第二阶段：Stuff合并（dedup=True）

全景分割的核心要求：
- **Thing**（实例类）：每个物体必须独立，如 car1, car2, car3
- **Stuff**（背景类）：同类必须合并，如 road, sky, grass

```python
def get_ids_area(masks, scores, dedup=False):
    # ... 生成m_id ...
    
    if dedup:
        # 合并同类stuff
        for equiv in stuff_equiv_classes.values():
            if len(equiv) > 1:
                for eq_id in equiv:
                    m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
```

**效果演示**：
```
合并前：
road -> mask0, mask4, mask7 (三个独立区域)

合并后：
m_id.masked_fill_(m_id.eq(4), 0)
m_id.masked_fill_(m_id.eq(7), 0)

结果：0, 4, 7 全部变成 0
```

### 第三阶段：ID转RGB

由于PNG只能保存RGB值，需要将segment ID映射为颜色：

```python
seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
```

**映射关系**：
```
segment id = 12345
    ⇔
RGB颜色 = (某个唯一颜色)
```

### 第四阶段：缩放到原图尺寸

```python
seg_img = seg_img.resize(
    size=(final_w, final_h), 
    resample=Image.NEAREST  # 必须用NEAREST!
)
```

**为什么不能用双线性插值？**

因为ID图是**离散标签**：
```
双线性插值：
像素A=0, 像素B=1
插值结果 = 0.37 → 无效ID！

NEAREST插值：
像素A=0, 像素B=1
插值结果 = 0或1 → 有效ID ✓
```

### 第五阶段：RGB转回ID

```python
np_seg_img = torch.ByteTensor(
    torch.ByteStorage.from_buffer(seg_img.tobytes())
).view(final_h, final_w, 3).numpy()

m_id = torch.from_numpy(rgb2id(np_seg_img))
# 恢复为 [final_h, final_w] 的ID图
```

### 第六阶段：计算面积

```python
area = []
for i in range(len(scores)):
    area.append(m_id.eq(i).sum().item())
```

**示例**：
```
m_id = [[0, 0, 1],
        [1, 1, 2]]

m_id.eq(0) = [[T, T, F],
              [F, F, F]] → area[0] = 2
m_id.eq(1) = [[F, F, T],
              [T, T, F]] → area[1] = 3
m_id.eq(2) = [[F, F, F],
              [F, F, T]] → area[2] = 1
```

### 第七阶段：删除小区域（核心！）

这是全景后处理中最精妙的部分：

```python
while True:
    filtered_small = torch.as_tensor(
        [area[i] <= 4 for i, c in enumerate(cur_classes)],
        dtype=torch.bool, device=keep.device
    )
    
    if filtered_small.any().item():
        # 删除小区域
        cur_scores = cur_scores[~filtered_small]
        cur_classes = cur_classes[~filtered_small]
        cur_masks = cur_masks[~filtered_small]
        
        # 重新竞争像素
        area, seg_img = get_ids_area(cur_masks, cur_scores)
    else:
        break
```

**为什么需要循环？**

```
初始状态：
segment0: 2000像素 (大)
segment1: 3像素 (小) ← 要被删除
segment2: 1500像素 (大)

删除segment1后，原来属于它的3个像素变成"无主之地"
重新执行argmax，这些像素会被分配给segment0或segment2

因此需要循环直到所有segment面积 > 4
```

### 第八阶段：生成COCO格式输出

```python
segments_info = []
for i, a in enumerate(area):
    cat = cur_classes[i].item()
    segments_info.append({
        "id": i,
        "isthing": self.is_thing_map[cat],
        "category_id": cat,
        "area": a
    })

predictions = {
    "png_string": out.getvalue(),  # PNG图像二进制
    "segments_info": segments_info  # 每个segment的元信息
}
```

## 五、实战案例解析

让我们通过一个真实的运行日志来理解整个过程：

```python
# 模型输出
out_logits: torch.Size([1, 100, 251])    # 1张图，100个queries，251个类别
raw_masks: torch.Size([1, 100, 267, 200]) # mask尺寸267x200
raw_boxes: torch.Size([1, 100, 4])
processed_sizes: tensor([[1066, 800]])   # 模型处理尺寸
target_sizes: tensor([[500, 375]])       # 原始图像尺寸
```

### Query过滤

```python
# 100个query中只有7个满足条件（置信度>0.85且非no-object）
cur_scores: torch.Size([7])   # 7个预测
cur_classes: torch.Size([7])
cur_masks: torch.Size([7, 1066, 800])  # 插值到1066×800
```

**关键观察**：100个query中只有7个检测到了有效物体，其余93个都是"no-object"或置信度太低。

### 生成ID图

```python
# cur_masks展平: [7, 852800] (1066*800=852800)
m_id = masks.transpose(0, 1).softmax(-1)  # [852800, 7]
m_id = m_id.argmax(-1).view(1066, 800)    # [1066, 800]
```

**实际输出的m_id**：
```python
tensor([[4, 4, 4, ..., 4, 4, 4],    # 前几行几乎全是4
        [4, 4, 4, ..., 4, 4, 4],
        [4, 4, 4, ..., 4, 4, 4],
        ...,
        [2, 2, 2, ..., 2, 2, 2],    # 后几行几乎全是2
        [2, 2, 2, ..., 2, 2, 2],
        [2, 2, 2, ..., 2, 2, 2]])
```

### 面积统计与过滤

假设统计结果：
```python
area = [150000, 80000, 200000, 50000, 300000, 3, 120000]
#        mask0   mask1   mask2   mask3   mask4   mask5  mask6
```

mask5只有3个像素，被认为是噪声，被删除并触发重新分配。

### 分辨率变换

```
1066×800 (模型处理尺寸)
    ↓ ID→RGB→NEAREST缩放→RGB→ID
500×375 (原始图像尺寸)
```

这个看似绕弯的操作，实际上是为了解决"离散标签无法插值"的核心问题。


## 六、关键技术要点总结

### 1. 像素分配核心
```python
m_id = masks.T.softmax(-1).argmax(-1)
```
这行代码完成了从mask logits到像素级分类的转换，是整个后处理的基础。

### 2. Stuff合并机制
通过`masked_fill`将同类stuff的多个mask合并为一个，确保全景分割的语义一致性。

### 3. 离散插值原则
必须使用`NEAREST`插值，不能用双线性，因为ID是离散标签而非连续值。

### 4. 循环删除策略
小区域删除后需要重新分配像素，直到所有区域面积>4，这是一个迭代收敛的过程。

### 5. 阈值过滤
置信度低于0.85的query被直接丢弃，减少了假阳性预测。

## 写在最后

DETR的后处理看似简单，实则暗藏玄机。特别是Panoptic后处理中的"删除小区域-重新分配"循环，体现了全景分割任务对区域完整性的严格要求。

理解这些后处理细节，不仅有助于调试模型，更能让你在部署DETR时游刃有余。从100个query到最终的全景分割结果，每一步都是精心设计的，共同构成了DETR完整的推理pipeline。

官方提供的三个Colab Notebook是很好的学习资源，建议结合本文的理解，动手实践一遍，相信你会有更深刻的体会。
