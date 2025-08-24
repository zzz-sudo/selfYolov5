#!/usr/bin/env python3
"""
LabelMe 转 YOLO 格式转换脚本使用示例.

本脚本展示了如何使用 labelme2yolo.py 进行数据格式转换
"""

import subprocess
import sys
from pathlib import Path


def run_conversion_example():
    """运行转换示例."""
    print("🚀 LabelMe 转 YOLO 格式转换示例")
    print("=" * 50)

    # 示例目录结构
    example_structure = """
    示例目录结构:
    
    my_labelme_dataset/
    ├── image1.jpg
    ├── image1.json
    ├── image2.jpg
    ├── image2.json
    ├── image3.png
    └── image3.json
    
    转换后的结构:
    
    output_dataset/
    ├── classes.txt
    ├── dataset.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
    """

    print(example_structure)

    # 检查转换脚本是否存在
    script_path = Path("labelme2yolo.py")
    if not script_path.exists():
        print("❌ 错误: 未找到 labelme2yolo.py 脚本")
        print("请确保脚本文件在当前目录中")
        return

    print("✅ 找到转换脚本: labelme2yolo.py")

    # 显示使用方法
    usage_examples = """
    使用方法示例:
    
    1. 基本转换:
       python labelme2yolo.py --input_dir /path/to/labelme/files --output_dir /path/to/output
    
    2. 指定类别文件:
       python labelme2yolo.py --input_dir /path/to/labelme/files --output_dir /path/to/output --classes classes.txt
    
    3. 自定义训练集/验证集比例:
       python labelme2yolo.py --input_dir /path/to/labelme/files --output_dir /path/to/output --train_ratio 0.8 --val_ratio 0.2
    
    4. 查看帮助信息:
       python labelme2yolo.py --help
    """

    print(usage_examples)

    # 显示脚本帮助信息
    print("📖 脚本帮助信息:")
    print("-" * 30)

    try:
        result = subprocess.run(
            [sys.executable, "labelme2yolo.py", "--help"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("获取帮助信息失败")
    except subprocess.TimeoutExpired:
        print("获取帮助信息超时")
    except Exception as e:
        print(f"获取帮助信息时出错: {e}")


def create_sample_classes_file():
    """创建示例类别文件."""
    classes_content = """person
car
dog
cat
bicycle
motorcycle
bus
truck
traffic_light
stop_sign
"""

    classes_file = Path("sample_classes.txt")
    with open(classes_file, "w", encoding="utf-8") as f:
        f.write(classes_content)

    print(f"✅ 创建示例类别文件: {classes_file}")
    print("类别列表:")
    for i, class_name in enumerate(classes_content.strip().split("\n")):
        print(f"  {i}: {class_name}")


def create_sample_dataset_yaml():
    """创建示例数据集配置文件."""
    yaml_content = """# 数据集配置文件示例
path: /path/to/your/dataset  # 数据集根目录
train: train/images          # 训练集图像目录
val: val/images              # 验证集图像目录
nc: 10                       # 类别数量
names:                       # 类别名称列表
  0: person
  1: car
  2: dog
  3: cat
  4: bicycle
  5: motorcycle
  6: bus
  7: truck
  8: traffic_light
  9: stop_sign
"""

    yaml_file = Path("sample_dataset.yaml")
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"✅ 创建示例数据集配置文件: {yaml_file}")


def show_training_commands():
    """显示训练命令示例."""
    training_commands = """
    🚀 YOLOv5 训练命令示例:
    
    1. 使用预训练权重训练:
       python train.py --data dataset.yaml --weights yolov5s.pt --img 640 --epochs 100
    
    2. 从头开始训练:
       python train.py --data dataset.yaml --weights '' --cfg yolov5s.yaml --img 640 --epochs 100
    
    3. 指定批次大小:
       python train.py --data dataset.yaml --weights yolov5s.pt --img 640 --epochs 100 --batch-size 16
    
    4. 多GPU训练:
       python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py \\
           --data dataset.yaml --weights yolov5s.pt --img 640 --epochs 100 --device 0,1,2,3
    
    5. 使用自定义超参数:
       python train.py --data dataset.yaml --weights yolov5s.pt --img 640 --epochs 100 --hyp hyp.yaml
    
    6. 启用混合精度训练:
       python train.py --data dataset.yaml --weights yolov5s.pt --img 640 --epochs 100 --amp
    """

    print(training_commands)


def show_validation_commands():
    """显示验证命令示例."""
    validation_commands = """
    🔍 模型验证命令示例:
    
    1. 验证最佳模型:
       python val.py --weights runs/train/exp1/weights/best.pt --data dataset.yaml --img 640
    
    2. 验证最后一轮模型:
       python val.py --weights runs/train/exp1/weights/last.pt --data dataset.yaml --img 640
    
    3. 保存验证结果:
       python val.py --weights runs/train/exp1/weights/best.pt --data dataset.yaml --img 640 --save-txt --save-conf
    
    4. 测试推理:
       python detect.py --weights runs/train/exp1/weights/best.pt --source test_image.jpg
    """

    print(validation_commands)


def main():
    """主函数."""
    print("🎯 LabelMe 数据集创建与 YOLOv5 训练完整指南")
    print("=" * 60)

    # 运行转换示例
    run_conversion_example()

    print("\n" + "=" * 60)

    # 创建示例文件
    print("📁 创建示例文件:")
    create_sample_classes_file()
    create_sample_dataset_yaml()

    print("\n" + "=" * 60)

    # 显示训练命令
    show_training_commands()

    print("\n" + "=" * 60)

    # 显示验证命令
    show_validation_commands()

    print("\n" + "=" * 60)

    # 总结
    summary = """
    📋 完整工作流程总结:
    
    1. 🏷️  使用 LabelMe 标注图像数据
    2. 🔄  使用 labelme2yolo.py 转换数据格式
    3. 🚀  使用 YOLOv5 训练自定义模型
    4. 🔍  验证模型性能
    5. 🎯  部署和推理
    
    💡 提示:
    - 确保标注质量，这是训练成功的关键
    - 合理设置训练参数，避免过拟合
    - 定期验证模型性能，及时调整策略
    - 保存最佳模型权重，用于后续部署
    """

    print(summary)

    print("🎉 示例完成！请根据您的实际需求调整参数和路径。")
    print("如有问题，请参考 README_LabelMe_Training.md 文件。")


if __name__ == "__main__":
    main()
