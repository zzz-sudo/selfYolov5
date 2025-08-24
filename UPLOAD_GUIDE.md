# 🚀 GitHub 上传指南

## 📋 上传前准备

### 1. 已排除的大文件

- ✅ 预训练权重文件 (_.pt, _.pth, \*.weights)
- ✅ 数据集文件 (datasets/ 目录)
- ✅ 训练结果 (runs/ 目录)
- ✅ 缓存文件 (_.cache, _.cache.npy)
- ✅ Python缓存文件 (**pycache**/)

### 2. 保留的重要文件

- ✅ 核心代码文件 (train.py, detect.py, segment/, classify/)
- ✅ 模型定义 (models/)
- ✅ 工具函数 (utils/)
- ✅ 配置文件 (data/\*.yaml)
- ✅ 自定义脚本 (labelme2yolo.py, labelme2yolo_seg.py, instance_segmentation_learning.py)
- ✅ 文档文件 (README.md, LICENSE, CITATION.cff)

## 🔧 上传步骤

### 步骤1: 初始化Git仓库

```bash
git init
git add .
git commit -m "Initial commit: YOLOv5 project with custom scripts"
```

### 步骤2: 添加远程仓库

```bash
git remote add origin https://github.com/zzz-sudo/selfYolov5.git
```

### 步骤3: 推送到GitHub

```bash
git branch -M main
git push -u origin main
```

## 📊 项目结构

```
selfYolov5/
├── 📁 models/              # 模型定义
├── 📁 utils/               # 工具函数
├── 📁 data/                # 数据集配置
├── 📁 segment/             # 实例分割模块
├── 📁 classify/            # 图像分类模块
├── 📄 train.py            # 目标检测训练
├── 📄 detect.py           # 目标检测推理
├── 📄 val.py              # 模型验证
├── 📄 export.py           # 模型导出
├── 📄 labelme2yolo.py    # LabelMe转YOLO格式
├── 📄 labelme2yolo_seg.py # LabelMe转实例分割格式
├── 📄 instance_segmentation_learning.py # 实例分割学习脚本
├── 📄 unified_training.py # 统一训练工作流程
├── 📄 README.md           # 项目说明文档
├── 📄 LICENSE             # 许可证
├── 📄 CITATION.cff        # 引用信息
└── 📄 .gitignore          # Git忽略文件
```

## ⚠️ 注意事项

1. **大文件已排除**: 预训练权重和数据集不会上传
2. **敏感信息**: 确保没有包含API密钥等敏感信息
3. **许可证**: 项目使用AGPL-3.0许可证
4. **依赖**: 用户需要安装requirements.txt中的依赖

## 🎯 项目特色

- 🎯 **完整的目标检测流程**: 从数据标注到模型训练
- ✂️ **实例分割支持**: 包含多边形标注和分割训练
- 🏷️ **图像分类功能**: 支持多类别分类任务
- 🛠️ **自定义脚本**: LabelMe数据转换和统一训练流程
- 📚 **详细文档**: 中文说明和操作指南
- 🔧 **环境问题解决**: 常见问题的解决方案

## 📞 联系方式

如有问题，请通过GitHub Issues联系。
