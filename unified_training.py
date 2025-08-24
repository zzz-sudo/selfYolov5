#!/usr/bin/env python3
"""
统一的YOLOv5训练工作流程脚本.

整合了数据标注、格式转换、模型训练、验证和推理的完整流程
完全禁用wandb，避免登录问题
"""

import os
import subprocess
import sys
from pathlib import Path


def disable_wandb():
    """完全禁用wandb日志记录."""
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"
    print("✅ 已完全禁用wandb日志记录")


def check_environment():
    """检查环境是否准备就绪."""
    print("🔍 检查环境...")

    # 检查LabelMe是否安装
    try:
        import labelme

        print("✅ LabelMe 已安装")
    except ImportError:
        print("❌ LabelMe 未安装，请运行: pip install labelme")
        return False

    # 检查YOLOv5环境
    if not Path("train.py").exists():
        print("❌ 请在YOLOv5项目根目录运行此脚本")
        return False

    print("✅ YOLOv5 环境正常")
    return True


def show_dataset_info():
    """显示数据集信息."""
    print("\n📁 数据集信息:")
    print("=" * 50)

    practice_dir = Path("datasets/labelme_practice")
    if practice_dir.exists():
        image_files = list(practice_dir.glob("*.jpg"))
        print(f"练习图像数量: {len(image_files)}")
        print("图像文件:")
        for img in image_files:
            print(f"  - {img.name}")

        # 检查是否有标注文件
        json_files = list(practice_dir.glob("*.json"))
        if json_files:
            print(f"\n已完成的标注文件: {len(json_files)}")
            for json_file in json_files:
                print(f"  - {json_file.name}")
        else:
            print("\n⚠️  还没有标注文件，请先使用LabelMe进行标注")
    else:
        print("❌ 练习数据集目录不存在")


def start_labelme():
    """启动LabelMe."""
    print("\n🚀 启动LabelMe...")
    print("=" * 50)

    practice_dir = Path("datasets/labelme_practice").absolute()
    print(f"标注目录: {practice_dir}")

    print("\n📋 标注步骤:")
    print("1. LabelMe将自动打开练习目录")
    print("2. 选择一张图像开始标注")
    print("3. 点击 'Create Rectangle' 创建标注框")
    print("4. 拖拽绘制边界框，完全包含目标物体")
    print("5. 输入类别名称 (如: person, car, dog)")
    print("6. 按 Ctrl+S 保存标注")
    print("7. 重复步骤3-6，标注所有目标物体")
    print("8. 选择下一张图像继续标注")

    print(f"\n💡 提示: 类别参考文件: {practice_dir}/classes.txt")

    # 启动LabelMe
    try:
        print(f"\n🎯 正在启动LabelMe，标注目录: {practice_dir}")
        subprocess.run(["labelme", str(practice_dir)], check=True)
    except subprocess.CalledProcessError:
        print("❌ 启动LabelMe失败")
        print("请手动运行: labelme datasets/labelme_practice")
    except FileNotFoundError:
        print("❌ 未找到LabelMe，请先安装: pip install labelme")


def convert_to_yolo():
    """转换为YOLO格式."""
    print("\n🔄 转换为YOLO格式...")
    print("=" * 50)

    input_dir = "datasets/labelme_practice"
    output_dir = "datasets/yolo_practice"

    if not Path(input_dir).exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return

    # 检查是否有标注文件
    json_files = list(Path(input_dir).glob("*.json"))
    if not json_files:
        print("❌ 没有找到标注文件，请先完成标注")
        return

    print(f"找到 {len(json_files)} 个标注文件")

    # 运行转换脚本
    try:
        cmd = [sys.executable, "labelme2yolo.py", "--input_dir", input_dir, "--output_dir", output_dir]

        print(f"运行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 转换完成!")
        print(result.stdout)

        # 显示转换结果
        if Path(output_dir).exists():
            print("\n📊 转换结果:")
            print(f"输出目录: {output_dir}")

            # 检查生成的文件
            for item in Path(output_dir).rglob("*"):
                if item.is_file():
                    print(f"  - {item.relative_to(output_dir)}")

    except subprocess.CalledProcessError as e:
        print(f"❌ 转换失败: {e}")
        print(f"错误输出: {e.stderr}")


def check_dataset():
    """检查数据集."""
    # 检查多个数据集选项
    datasets = {"COCO128": "data/coco128_custom.yaml", "LabelMe练习集": "datasets/yolo_practice/dataset.yaml"}

    available_datasets = []
    for name, path in datasets.items():
        if Path(path).exists():
            available_datasets.append((name, path))

    if not available_datasets:
        print("❌ 未找到可用的数据集配置")
        print("请选择以下选项之一:")
        print("1. 使用COCO128数据集: 直接可用")
        print("2. 使用LabelMe练习集: 需要先运行数据转换")
        return False

    print("✅ 可用的数据集:")
    for i, (name, path) in enumerate(available_datasets, 1):
        print(f"  {i}. {name}: {path}")

    # 默认使用第一个可用数据集
    return available_datasets[0][1]


def start_training():
    """开始训练."""
    print("\n🚀 开始训练自定义模型...")
    print("=" * 50)

    # 检查并选择数据集
    dataset_yaml = check_dataset()
    if not dataset_yaml:
        return False

    # 完全禁用wandb
    disable_wandb()

    # 根据数据集选择项目名称
    if "coco128" in dataset_yaml:
        project_name = "coco128_training"
        exp_name = "yolov5s_coco128"
    else:
        project_name = "custom_training"
        exp_name = "labelme_practice"

    # 训练命令
    cmd = [
        sys.executable,
        "train.py",
        "--data",
        dataset_yaml,
        "--weights",
        "yolov5s.pt",
        "--img",
        "640",
        "--epochs",
        "30",  # 适中的训练轮次
        "--batch-size",
        "8",  # 较小的批次大小
        "--project",
        project_name,  # 自定义项目名称
        "--name",
        exp_name,  # 自定义实验名称
    ]

    print(f"训练命令: {' '.join(cmd)}")
    print("\n⏳ 训练开始，请耐心等待...")
    print("💡 提示: 训练过程中会显示损失曲线和进度")

    try:
        # 运行训练
        subprocess.run(cmd, check=True)
        print("\n🎉 训练完成!")
        print("结果保存在: runs/train/custom_training/labelme_practice/")

        # 显示结果位置
        result_dir = Path("runs/train/custom_training/labelme_practice")
        if result_dir.exists():
            print("\n📁 训练结果:")
            for item in result_dir.rglob("*"):
                if item.is_file():
                    print(f"  - {item.relative_to(result_dir)}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        print("请检查错误信息并重试")
        return False

    return True


def validate_model():
    """验证模型性能..."""
    print("\n🔍 验证模型性能...")
    print("=" * 50)

    # 检查并选择数据集
    dataset_yaml = check_dataset()
    if not dataset_yaml:
        return

    # 查找最佳模型权重 (检查多个可能的路径)
    possible_paths = [
        "runs/train/custom_training/labelme_practice/weights",
        "runs/train/coco128_training/yolov5s_coco128/weights",
    ]

    weights_dir = None
    for path in possible_paths:
        if Path(path).exists():
            weights_dir = Path(path)
            break

    if not weights_dir:
        print("❌ 未找到训练结果，请先完成训练")
        return

    best_weights = weights_dir / "best.pt"
    if not best_weights.exists():
        print("❌ 未找到最佳模型权重")
        return

    print(f"✅ 找到最佳模型: {best_weights}")

    # 验证命令
    cmd = [sys.executable, "val.py", "--weights", str(best_weights), "--data", dataset_yaml, "--img", "640"]

    print(f"验证命令: {' '.join(cmd)}")
    print("\n⏳ 开始验证...")

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ 验证完成!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 验证失败: {e}")


def test_inference():
    """测试推理."""
    print("\n🎯 测试模型推理...")
    print("=" * 50)

    # 检查并选择数据集
    dataset_yaml = check_dataset()
    if not dataset_yaml:
        return

    # 查找最佳模型权重 (检查多个可能的路径)
    possible_paths = [
        "runs/train/custom_training/labelme_practice/weights",
        "runs/train/coco128_training/yolov5s_coco128/weights",
    ]

    weights_dir = None
    for path in possible_paths:
        if Path(path).exists():
            weights_dir = Path(path)
            break

    if not weights_dir:
        print("❌ 未找到训练结果，请先完成训练")
        return

    best_weights = weights_dir / "best.pt"
    if not best_weights.exists():
        print("❌ 未找到最佳模型权重")
        return

    print(f"✅ 使用模型: {best_weights}")

    # 根据数据集选择测试图像
    if "coco128" in dataset_yaml:
        test_image = "datasets/coco128/images/train2017/000000000009.jpg"
        project_name = "coco128_inference"
        exp_name = "yolov5s_coco128"
    else:
        test_image = "datasets/labelme_practice/000000000009.jpg"
        project_name = "custom_inference"
        exp_name = "labelme_practice"

    if not Path(test_image).exists():
        print(f"❌ 测试图像不存在: {test_image}")
        return

    # 推理命令
    cmd = [
        sys.executable,
        "detect.py",
        "--weights",
        str(best_weights),
        "--source",
        test_image,
        "--project",
        project_name,
        "--name",
        exp_name,
    ]

    print(f"推理命令: {' '.join(cmd)}")
    print("\n⏳ 开始推理...")

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ 推理完成!")

        # 显示结果位置
        result_dir = Path("runs/detect/custom_inference/labelme_practice")
        if result_dir.exists():
            print(f"\n📁 推理结果保存在: {result_dir}")
            for item in result_dir.glob("*"):
                if item.is_file():
                    print(f"  - {item.name}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 推理失败: {e}")


def show_complete_workflow():
    """显示完整工作流程."""
    print("\n📋 完整工作流程总结:")
    print("=" * 50)

    workflow = """
    1. 🏷️  数据标注 (LabelMe)
       - 启动LabelMe: labelme datasets/labelme_practice
       - 标注5张练习图像
       - 保存为JSON格式
    
    2. 🔄  格式转换
       - 运行: python labelme2yolo.py --input_dir datasets/labelme_practice --output_dir datasets/yolo_practice
       - 自动生成YOLO格式数据集
    
    3. 🚀  模型训练
       - 运行: python train.py --data datasets/yolo_practice/dataset.yaml --weights yolov5s.pt --img 640 --epochs 30
       - 训练自定义模型
    
    4. 🔍  模型验证
       - 运行: python val.py --weights runs/train/custom_training/labelme_practice/weights/best.pt --data datasets/yolo_practice/dataset.yaml
       - 验证模型性能
    
    5. 🎯  模型推理
       - 运行: python detect.py --weights runs/train/custom_training/labelme_practice/weights/best.pt --source test_image.jpg
       - 测试模型效果
    """

    print(workflow)


def show_workflow():
    """显示当前工作流程状态."""
    print("\n📋 当前工作流程状态:")
    print("=" * 30)

    # 检查当前使用的数据集
    current_dataset = check_dataset()
    if current_dataset:
        print(f"当前使用数据集: {current_dataset}")

    workflow = """
    1. 🏷️  数据标注
       - 状态: {'✅ 已完成' if Path('datasets/labelme_practice').glob('*.json') else '⏳ 待完成'}
       - 使用LabelMe标注5张图像
       - 生成JSON格式标注文件
    
    2. 🔄  格式转换
       - 状态: {'✅ 已完成' if Path('datasets/yolo_practice/dataset.yaml').exists() else '⏳ 待完成'}
       - 转换为YOLO格式
       - 自动分割训练集/验证集
    
    3. 🚀  模型训练
       - 状态: {'✅ 已完成' if Path('runs/train/custom_training/labelme_practice/weights/best.pt').exists() or Path('runs/train/coco128_training/yolov5s_coco128/weights/best.pt').exists() else '⏳ 待完成'}
       - 使用预训练权重yolov5s.pt
       - 训练30个epochs
       - 结果保存在runs/train/目录下
    
    4. 🔍  模型验证
       - 状态: {'✅ 已完成' if Path('runs/train/custom_training/labelme_practice/weights/best.pt').exists() or Path('runs/train/coco128_training/yolov5s_coco128/weights/best.pt').exists() else '⏳ 待完成'}
       - 验证模型在验证集上的性能
       - 生成mAP等评估指标
    
    5. 🎯  模型推理
       - 状态: {'✅ 已完成' if Path('runs/detect/custom_inference/labelme_practice').exists() or Path('runs/detect/coco128_inference/yolov5s_coco128').exists() else '⏳ 待完成'}
       - 使用训练好的模型进行预测
       - 测试单张图像或视频
    """

    print(workflow)


def select_dataset():
    """选择数据集."""
    print("\n🔧 数据集选择")
    print("=" * 30)

    datasets = {
        "1": ("COCO128", "data/coco128_custom.yaml", "使用预标注的COCO128数据集 (推荐)"),
        "2": ("LabelMe练习集", "datasets/yolo_practice/dataset.yaml", "使用LabelMe标注的练习数据集"),
    }

    print("可用的数据集:")
    for key, (name, path, desc) in datasets.items():
        status = "✅ 可用" if Path(path).exists() else "❌ 不可用"
        print(f"  {key}. {name}: {status}")
        print(f"     {desc}")

    try:
        choice = input("\n请选择数据集 (1-2): ").strip()
        if choice in datasets:
            name, path, desc = datasets[choice]
            if Path(path).exists():
                print(f"✅ 已选择数据集: {name}")
                print(f"配置文件: {path}")
                print("现在可以使用训练、验证、推理等功能")
            else:
                print(f"❌ 数据集 {name} 不可用")
                if choice == "2":
                    print("请先完成LabelMe标注和数据转换")
        else:
            print("❌ 无效选择")
    except KeyboardInterrupt:
        print("\n⏹️  已取消")


def main():
    """主函数."""
    print("🎯 统一的YOLOv5训练工作流程")
    print("=" * 60)

    # 检查环境
    if not check_environment():
        return

    while True:
        print("\n📋 请选择操作:")
        print("1. 📁 查看数据集信息")
        print("2. 🏷️  启动LabelMe进行标注")
        print("3. 🔄 转换为YOLO格式")
        print("4. 🚀 开始训练模型")
        print("5. 🔍 验证模型性能")
        print("6. 🎯 测试模型推理")
        print("7. 📋 查看完整工作流程")
        print("8. 📊 查看当前状态")
        print("9. 🔧 选择数据集")
        print("10. 🚪 退出")

        try:
            choice = input("\n请输入选择 (1-9): ").strip()

            if choice == "1":
                show_dataset_info()
            elif choice == "2":
                start_labelme()
            elif choice == "3":
                convert_to_yolo()
            elif choice == "4":
                if check_dataset():
                    start_training()
                else:
                    print("❌ 请先完成数据转换")
            elif choice == "5":
                validate_model()
            elif choice == "6":
                test_inference()
            elif choice == "7":
                show_complete_workflow()
            elif choice == "8":
                show_workflow()
            elif choice == "9":
                select_dataset()
            elif choice == "10":
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请输入1-10")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    main()
