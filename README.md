# 井盖隐患智能识别系统 (Manhole Cover Defect Detection System)

本项目旨在利用深度学习技术，实现对城市井盖状态的智能识别与检测。系统能够自动识别井盖的多种隐患类型，包括破损、缺失、未盖好、井圈问题等，为城市市政维护提供技术支持。

## 1. 项目概述

本项目基于 **YOLOv8** 目标检测算法，针对井盖检测场景进行了定制化的数据处理和模型训练。

### 支持检测的类别
*   **broke**: 井盖破损
*   **lose**: 井盖缺失
*   **uncovered**: 井盖未盖好（移位）
*   **circle**: 井圈问题（井盖周围路面破损）
*   **good**: 完好井盖

## 2. 技术方案

### 2.1 数据增强与图像增强 (Data Augmentation)

为了解决原始数据量可能不足以及提升模型在复杂环境下的鲁棒性，本项目采用了 **Albumentations** 库进行离线数据增强。

**增强策略：**
我们在训练前对每张原始图片生成 **2张** 额外的增强图片，将训练集规模扩充为原来的 **3倍**。

**具体手段：**

1.  **几何变换 (Geometric Transformations)**: 模拟不同的拍摄角度和距离。
    *   **随机水平翻转 (HorizontalFlip)**: 概率 0.5。
    *   **随机旋转 (Rotate)**: 旋转角度限制在 ±15°，概率 0.5。
    *   **随机缩放 (RandomScale)**: 缩放比例限制在 ±10%，概率 0.5。

2.  **像素级图像增强 (Pixel-level Enhancements)**: 模拟不同的光照、天气和成像质量。
    *   **CLAHE (自适应直方图均衡化)**: 限制对比度自适应直方图均衡化，有效增强图像局部对比度，突出井盖纹理细节。
    *   **随机亮度与对比度 (RandomBrightnessContrast)**: 模拟不同时间段的光照变化。
    *   **色调/饱和度/明度变换 (HueSaturationValue)**: 模拟不同摄像头的色彩偏差。
    *   **高斯噪声 (GaussNoise)**: 模拟低光照下的传感器噪点。
    *   **模糊 (Blur)**: 模拟运动模糊或对焦不准的情况。

### 2.2 模型结构与参数设置

本项目选用 **Ultralytics YOLOv8** 作为核心检测模型。

**模型选择：**
*   **模型版本**: **YOLOv8s (Small)**
*   **选择理由**: 考虑到部署环境（如 RTX 4060 Laptop）的性能限制，YOLOv8s 在检测精度和推理速度之间取得了最佳平衡。相比 Nano 版本精度更高，相比 Medium/Large 版本计算量更小，适合实时性要求较高的边缘侧部署。

**训练参数设置：**

| 参数 | 设置值 | 说明 |
| :--- | :--- | :--- |
| **Epochs** | 100 | 最大训练轮数，配合 Early Stopping 防止过拟合。 |
| **Batch Size** | 16 | 针对 8GB 显存优化，确保显存利用率最大化且不溢出。 |
| **Image Size** | 640 | 标准输入分辨率，平衡细节保留与计算量。 |
| **Patience** | 20 | 早停机制，如果验证集精度 20 轮不提升则提前结束。 |
| **Optimizer** | Auto | 自动选择优化器（通常为 SGD 或 AdamW）。 |
| **LR Scheduler** | Cosine | 余弦退火学习率调度 (Cosine Annealing)，有助于模型收敛到更优解。 |
| **AMP** | True | 开启自动混合精度训练 (Automatic Mixed Precision)，减少显存占用并加速训练。 |

## 3. 快速开始

### 环境依赖
```bash
pip install -r requirements.txt
```

### 数据准备
将原始数据放入 `data/` 目录，运行以下命令进行数据清洗、增强和划分：
```bash
python main.py
```
*(注：`main.py` 会自动调用 `pre.py` 进行数据处理，随后开始训练)*

### 模型评估
训练完成后，评估模型性能并生成 PR 曲线、混淆矩阵等：
```bash
python evaluate.py
```

### 推理检测
使用训练好的模型对新图片进行测试：
```bash
# 检测单张图片
python process.py test/1.webp

# 检测文件夹
python process.py test/
```
