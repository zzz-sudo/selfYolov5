# YOLOv5 目标检测、实例分割和图像分类项目

<div align="center">
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="YOLOv5 Banner" width="100%">
</div>

## 🚀 快速导航

| 需求 | 查看章节 | 快速链接 |
|------|----------|----------|
| 🆕 **新手入门** | [快速开始](#-快速开始) | 环境安装、练习数据集 |
| 🏷️ **数据标注** | [LabelMe 数据标注完整指南](#-labelme-数据标注完整指南) | 安装LabelMe、标注步骤、格式转换 |
| 🚀 **模型训练** | [YOLOv5 训练流程](#-yolov5-训练流程) | 训练准备、命令详解、过程监控 |
| 🔍 **模型验证** | [模型验证与测试](#-模型验证与测试) | 验证命令、性能指标、推理测试 |
| ⚠️ **问题解决** | [常见环境问题解决](#️-常见环境问题解决) | numpy兼容、依赖冲突、wandb问题 |
| 🔧 **高级功能** | [高级功能](#-高级功能) | 模型导出、超参数优化、部署指南 |
| 📊 **性能参考** | [性能指标](#-性能指标) | 各模型性能对比表 |
| 🛠️ **开发指南** | [开发指南](#️-开发指南) | 项目结构、自定义数据集 |

## 📖 项目简介

YOLOv5 是由 Ultralytics 开发的最先进的计算机视觉模型，基于 PyTorch 框架构建。该项目支持三种主要的计算机视觉任务：

- 🎯 **目标检测 (Object Detection)**: 检测图像中的物体位置和类别
- ✂️ **实例分割 (Instance Segmentation)**: 为每个检测到的物体生成像素级掩码
- 🏷️ **图像分类 (Image Classification)**: 对图像进行类别分类

## ✨ 主要特性

- **高性能**: 在 COCO 数据集上达到最先进的性能
- **易用性**: 简单的命令行接口和 Python API
- **多任务支持**: 检测、分割、分类三合一
- **多平台部署**: 支持 PyTorch、ONNX、TensorRT、CoreML 等格式
- **实时推理**: 优化的推理速度，适合实时应用
- **自动下载**: 预训练模型和数据集自动下载

## 🚀 快速开始

### 环境要求

- Python >= 3.8.0
- PyTorch >= 1.8.0
- CUDA >= 10.2 (GPU 训练推荐)

### 🎯 练习数据集

我们为您准备了完整的练习数据集，包含：

1. **COCO128数据集**: 128张图像，适合大规模训练
2. **LabelMe练习集**: 5张精选图像，适合入门练习
3. **完整工作流程**: 从标注到训练的完整示例

#### 快速体验完整流程

```bash
# 1. 运行统一训练工作流程 (推荐)
python unified_training.py

# 2. 手动执行各个步骤
# 标注数据
labelme datasets/labelme_practice

# 转换格式
python labelme2yolo.py --input_dir datasets/labelme_practice --output_dir datasets/yolo_practice

# 训练模型
python train.py --data datasets/yolo_practice/dataset.yaml --weights yolov5s.pt --img 640 --epochs 30
```

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/ultralytics/yolov5
cd yolov5

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装LabelMe (用于数据标注)
pip install labelme
```

## 🏷️ LabelMe 数据标注完整指南

### 什么是LabelMe？

LabelMe 是一个开源的图像标注工具，专门用于创建计算机视觉数据集。它支持多种标注格式，包括矩形框、多边形、线条等。

### 安装LabelMe

```bash
# 使用pip安装
pip install labelme

# 或者使用conda安装
conda install -c conda-forge labelme
```

### 启动LabelMe

```bash
# 启动LabelMe并指定工作目录
labelme datasets/labelme_practice

# 或者在当前目录启动
labelme
```

### 标注步骤详解

#### 1. 基本操作流程

```bash
1. 打开图像文件
2. 选择标注工具 (Create Rectangle)
3. 绘制边界框
4. 输入类别标签
5. 保存标注 (Ctrl+S)
6. 重复步骤2-5标注所有目标
7. 选择下一张图像继续
```

#### 2. 标注技巧

- **边界框绘制**: 确保完全包含目标物体，避免过小或过大
- **标签命名**: 使用一致的类别名称，避免拼写错误
- **质量控制**: 多角度检查标注质量，确保准确性

#### 3. 推荐类别

根据COCO数据集，建议使用以下类别：
```
person, car, dog, cat, bicycle, motorcycle, bus, truck, 
traffic_light, stop_sign, chair, table, bottle, cup, 
fork, knife, spoon, bowl, banana, apple, sandwich, 
orange, broccoli, carrot, hot_dog, pizza, donut, cake
```

### 数据格式转换

#### 使用转换脚本

我们提供了专门的转换脚本 `labelme2yolo.py`：

```bash
# 基本用法
python labelme2yolo.py --input_dir datasets/labelme_practice --output_dir datasets/yolo_practice

# 指定类别文件
python labelme2yolo.py --input_dir datasets/labelme_practice --output_dir datasets/yolo_practice --classes classes.txt

# 自定义训练集/验证集比例
python labelme2yolo.py --input_dir datasets/labelme_practice --output_dir datasets/yolo_practice --train_ratio 0.8 --val_ratio 0.2
```

#### 转换脚本功能

```python
# 主要功能：
1. 读取LabelMe JSON标注文件
2. 转换为YOLO格式标签文件
3. 自动分割训练集和验证集
4. 生成数据集配置文件 dataset.yaml
5. 统计标注信息
```

#### 输出目录结构

```
datasets/yolo_practice/
├── classes.txt              # 类别列表
├── dataset.yaml             # 数据集配置文件
├── images/                  # 所有图像
├── labels/                  # 所有标签
├── train/                   # 训练集
│   ├── images/             # 训练图像
│   └── labels/             # 训练标签
└── val/                     # 验证集
    ├── images/              # 验证图像
    └── labels/              # 验证标签
```

### 数据集配置文件

转换完成后，会自动生成 `dataset.yaml` 文件：

```yaml
# 数据集配置
path: F:/PythonLearning/yolov5/datasets/yolo_practice
train: train/images
val: val/images
nc: 4  # 类别数量
names: ['cat', 'food', 'person', 'truck']  # 类别名称
```

## 🚀 YOLOv5 训练流程

### 训练前准备

#### 1. 检查数据集

```bash
# 确保数据集配置正确
python -c "
import yaml
with open('datasets/yolo_practice/dataset.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f'类别数量: {config[\"nc\"]}')
print(f'类别名称: {config[\"names\"]}')
print(f'训练集路径: {config[\"train\"]}')
print(f'验证集路径: {config[\"val\"]}')
"
```

#### 2. 选择预训练模型

```bash
# 推荐使用预训练模型 (适合小数据集)
python train.py --data datasets/yolo_practice/dataset.yaml --weights yolov5s.pt --img 640 --epochs 30

# 从头开始训练 (需要大量数据和计算资源)
python train.py --data datasets/yolo_practice/dataset.yaml --weights '' --cfg yolov5s.yaml --img 640 --epochs 100
```

### 训练命令详解

#### 基本训练命令

```bash
# 单GPU训练
python train.py \
    --data datasets/yolo_practice/dataset.yaml \
    --weights yolov5s.pt \
    --img 640 \
    --epochs 30 \
    --batch-size 8 \
    --project custom_training \
    --name labelme_practice
```

#### 高级训练选项

```bash
# 多GPU训练
python -m torch.distributed.run --nproc_per_node 2 train.py \
    --data datasets/yolo_practice/dataset.yaml \
    --weights yolov5s.pt \
    --img 640 \
    --epochs 30 \
    --batch-size 16

# 自定义超参数
python train.py \
    --data datasets/yolo_practice/dataset.yaml \
    --weights yolov5s.pt \
    --img 640 \
    --epochs 30 \
    --batch-size 8 \
    --lr0 0.001 \
    --momentum 0.937 \
    --weight_decay 0.0005
```

### 训练过程监控

#### 1. 实时监控

训练过程中会显示：
- 损失曲线 (box_loss, obj_loss, cls_loss)
- GPU内存使用情况
- 训练进度和速度
- 验证集性能指标

#### 2. TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir runs/train

# 在浏览器中查看
# http://localhost:6006
```

#### 3. 训练结果

训练完成后，结果保存在：
```
runs/train/custom_training/labelme_practice/
├── weights/
│   ├── best.pt          # 最佳模型权重
│   └── last.pt          # 最后一轮模型权重
├── results.png           # 训练结果图表
├── confusion_matrix.png  # 混淆矩阵
├── labels.jpg           # 标签分布图
└── train_batch0.jpg     # 训练批次示例
```

## 🔍 模型验证与测试

### 模型验证

#### 1. 验证命令

```bash
# 验证最佳模型
python val.py \
    --weights runs/train/custom_training/labelme_practice/weights/best.pt \
    --data datasets/yolo_practice/dataset.yaml \
    --img 640

# 验证最后一轮模型
python val.py \
    --weights runs/train/custom_training/labelme_practice/weights/last.pt \
    --data datasets/yolo_practice/dataset.yaml \
    --img 640
```

#### 2. 验证指标

验证完成后会显示：
- **mAP@0.5**: 交并比阈值为0.5时的平均精度
- **mAP@0.5:0.95**: 交并比阈值从0.5到0.95的平均精度
- **Precision**: 精确率
- **Recall**: 召回率
- **各类别性能**: 每个类别的具体表现

### 模型推理测试

#### 1. 单张图像推理

```bash
# 使用训练好的模型进行推理
python detect.py \
    --weights runs/train/custom_training/labelme_practice/weights/best.pt \
    --source datasets/labelme_practice/000000000009.jpg \
    --project custom_inference \
    --name labelme_practice
```

#### 2. 批量推理

```bash
# 推理整个目录
python detect.py \
    --weights runs/train/custom_training/labelme_practice/weights/best.pt \
    --source datasets/labelme_practice/ \
    --project custom_inference \
    --name labelme_practice
```

#### 3. 推理结果

推理完成后，结果保存在：
```
runs/detect/custom_inference/labelme_practice/
├── 000000000009.jpg     # 带检测框的图像
├── 000000000025.jpg     # 带检测框的图像
└── ...
```

## ⚠️ 常见环境问题解决

### 问题1: numpy 版本不兼容

如果遇到 `ValueError: numpy.dtype size changed` 错误：

```bash
# 降级 numpy 到兼容版本
pip install "numpy<2.0.0,>=1.23.5"

# 或者重新安装 pandas
pip uninstall pandas
pip install pandas
```

**原因分析**: 这是因为 `numpy 2.0+` 与 `pandas 1.3.5` 存在二进制不兼容问题。

### 问题2: 依赖冲突

如果遇到包依赖冲突：

```bash
# 创建虚拟环境
python -m venv yolov5_env
yolov5_env\Scripts\activate  # Windows
# source yolov5_env/bin/activate  # Linux/Mac

# 在虚拟环境中安装
pip install -r requirements.txt
```

### 问题3: CUDA 相关问题

确保 PyTorch 版本与 CUDA 版本匹配：

```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch
# 访问 https://pytorch.org/get-started/locally/ 选择正确版本
```

### 问题4: wandb 登录问题

如果遇到 wandb 登录问题：

```bash
# 方法1: 禁用wandb
export WANDB_DISABLED=true

# 方法2: 在训练命令中添加环境变量
WANDB_DISABLED=true python train.py --data dataset.yaml --weights yolov5s.pt

# 方法3: 使用我们的统一训练脚本 (已自动禁用wandb)
python unified_training.py
```

## 🔧 高级功能

### 1. 模型导出

```bash
# 导出为 ONNX 格式
python export.py --weights yolov5s.pt --include onnx

# 导出为 TensorRT 格式
python export.py --weights yolov5s.pt --include engine --device 0

# 导出为 CoreML 格式
python export.py --weights yolov5s.pt --include coreml
```

### 2. 超参数优化

```bash
# 使用遗传算法优化超参数
python train.py --data coco.yaml --weights yolov5s.pt --evolve
```

### 3. 测试时增强 (TTA)

```bash
# 启用测试时增强
python val.py --weights yolov5s.pt --data coco.yaml --augment
```

### 4. 模型集成

```bash
# 集成多个模型
python val.py --weights yolov5s.pt yolov5m.pt yolov5l.pt --data coco.yaml
```

## 📊 性能指标

### 目标检测性能 (COCO 数据集)

| 模型 | 输入尺寸 | mAP@0.5:0.95 | mAP@0.5 | 参数量(M) | FLOPs(B) |
|------|----------|---------------|---------|------------|-----------|
| YOLOv5n | 640 | 28.0 | 45.7 | 1.9 | 4.5 |
| YOLOv5s | 640 | 37.4 | 56.8 | 7.2 | 16.5 |
| YOLOv5m | 640 | 45.4 | 64.1 | 21.2 | 49.0 |
| YOLOv5l | 640 | 49.0 | 67.3 | 46.5 | 109.1 |
| YOLOv5x | 640 | 50.7 | 68.9 | 86.7 | 205.7 |

### 实例分割性能

| 模型 | 输入尺寸 | mAP@0.5:0.95 (box) | mAP@0.5:0.95 (mask) | 参数量(M) |
|------|----------|---------------------|---------------------|------------|
| YOLOv5n-seg | 640 | 27.6 | 23.4 | 2.0 |
| YOLOv5s-seg | 640 | 37.6 | 31.7 | 7.6 |
| YOLOv5m-seg | 640 | 45.0 | 37.1 | 22.0 |
| YOLOv5l-seg | 640 | 49.0 | 39.9 | 47.9 |
| YOLOv5x-seg | 640 | 50.7 | 41.4 | 88.8 |

### 图像分类性能 (ImageNet)

| 模型 | 输入尺寸 | Top-1 Acc | Top-5 Acc | 参数量(M) |
|------|----------|------------|------------|------------|
| YOLOv5n-cls | 224 | 64.6 | 85.4 | 2.5 |
| YOLOv5s-cls | 224 | 71.5 | 90.2 | 5.4 |
| YOLOv5m-cls | 224 | 75.9 | 92.9 | 12.9 |
| YOLOv5l-cls | 224 | 78.0 | 94.0 | 26.5 |
| YOLOv5x-cls | 224 | 79.0 | 94.4 | 48.1 |

## 🛠️ 开发指南

### 项目结构

```
yolov5/
├── models/              # 模型定义
│   ├── yolo.py         # YOLO 模型主文件
│   ├── common.py       # 通用网络层
│   └── experimental.py # 实验性模块
├── utils/               # 工具函数
│   ├── dataloaders.py  # 数据加载器
│   ├── loss.py         # 损失函数
│   ├── metrics.py      # 评估指标
│   └── plots.py        # 可视化工具
├── data/                # 数据集配置
├── train.py            # 目标检测训练
├── detect.py           # 目标检测推理
├── segment/            # 实例分割模块
│   ├── train.py       # 分割训练
│   ├── predict.py     # 分割推理
│   └── val.py         # 分割验证
├── classify/           # 图像分类模块
│   ├── train.py       # 分类训练
│   ├── predict.py     # 分类推理
│   └── val.py         # 分类验证
├── datasets/           # 练习数据集
│   ├── coco128/       # COCO128数据集
│   └── labelme_practice/ # LabelMe练习集
├── labelme2yolo.py    # LabelMe转YOLO格式脚本
├── unified_training.py # 统一训练工作流程脚本
└── README.md           # 项目说明文档
```

### 自定义数据集

#### 1. 目标检测数据集

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

标签格式 (YOLO):
```
class_id x_center y_center width height
```

#### 2. 实例分割数据集

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

标签格式:
```
class_id x1 y1 x2 y2 ... xn yn
```

#### 3. 图像分类数据集

```
dataset/
├── class1/
│   ├── image1.jpg
│   └── image2.jpg
├── class2/
│   ├── image3.jpg
│   └── image4.jpg
└── ...
```

### 训练配置

创建 `hyp.yaml` 文件：

```yaml
# 学习率
lr0: 0.01
lrf: 0.1

# 动量
momentum: 0.937
weight_decay: 0.0005

# 数据增强
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
```

## 🚀 部署指南

### 1. PyTorch 部署

```python
import torch

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 推理
img = 'path/to/image.jpg'
results = model(img)

# 处理结果
results.print()
results.show()
```

### 2. ONNX 部署

```python
import onnxruntime as ort

# 加载 ONNX 模型
session = ort.InferenceSession('yolov5s.onnx')

# 预处理输入
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# 推理
outputs = session.run(output_names, {input_name: input_data})
```

### 3. TensorRT 部署

```python
import tensorrt as trt
import pycuda.driver as cuda

# 加载 TensorRT 引擎
with open('yolov5s.engine', 'rb') as f:
    engine_data = f.read()
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)

# 创建执行上下文
context = engine.create_execution_context()
```

## 🔍 故障排除

### 常见问题

1. **CUDA 内存不足**
```bash
# 减少批次大小
python train.py --batch-size 16

# 使用混合精度训练
python train.py --amp
```

2. **数据集加载错误**
```bash
# 检查数据集路径
python train.py --data custom.yaml --img 640 --epochs 1
```

3. **模型导出失败**
```bash
# 检查依赖安装
pip install onnx onnxsim
```

### 性能优化

1. **训练优化**
   - 使用多GPU训练
   - 启用混合精度训练
   - 优化数据加载器

2. **推理优化**
   - 使用 TensorRT 加速
   - 启用半精度推理
   - 批处理推理

## 📚 参考资料

- [YOLOv5 官方文档](https://docs.ultralytics.com/yolov5/)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [COCO 数据集](https://cocodataset.org/)
- [ImageNet 数据集](https://image-net.org/)
- [LabelMe 官方文档](https://github.com/wkentaro/labelme)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 AGPL-3.0 许可证。详情请查看 [LICENSE](LICENSE) 文件。

## 📞 联系方式

- GitHub Issues: [项目问题反馈](https://github.com/ultralytics/yolov5/issues)
- 官方文档: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- 社区讨论: [Discord](https://discord.com/invite/ultralytics)

---

<div align="center">
  <p>如果这个项目对您有帮助，请给个 ⭐️ 支持一下！</p>
</div>
