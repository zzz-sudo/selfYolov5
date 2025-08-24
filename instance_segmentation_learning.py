#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5 实例分割完整学习流程

包含：
1. 数据集准备和下载
2. LabelMe多边形标注
3. 实例分割模型训练
4. 模型验证和推理
5. 详细的操作指导

"""

import os
import sys
import subprocess
import zipfile
import requests
from pathlib import Path
import shutil


class InstanceSegmentationLearning:
    def __init__(self):
        self.project_root = Path.cwd()
        self.datasets_dir = self.project_root / "datasets"
        self.coco128_seg_dir = self.datasets_dir / "coco128-seg"
        self.practice_dir = self.datasets_dir / "segmentation_practice"
        self.output_dir = self.datasets_dir / "yolo_seg_practice"
        
        # 确保目录存在
        self.datasets_dir.mkdir(exist_ok=True)
        self.practice_dir.mkdir(exist_ok=True)
        
    def print_header(self, title):
        """打印标题"""
        print("\n" + "="*60)
        print(f"🎯 {title}")
        print("="*60)
    
    def print_step(self, step_num, title):
        """打印步骤标题"""
        print(f"\n📋 步骤 {step_num}: {title}")
        print("-" * 40)
    
    def check_environment(self):
        """检查环境"""
        self.print_header("环境检查")
        
        # 检查YOLOv5环境
        if not (self.project_root / "train.py").exists():
            print("❌ 请在YOLOv5项目根目录运行此脚本")
            return False
        
        # 检查实例分割模块
        if not (self.project_root / "segment" / "train.py").exists():
            print("❌ 未找到实例分割模块，请确保使用支持实例分割的YOLOv5版本")
            return False
        
        # 检查LabelMe
        try:
            import labelme
            print("✅ LabelMe 已安装")
        except ImportError:
            print("❌ LabelMe 未安装，请运行: pip install labelme")
            return False
        
        print("✅ 环境检查通过")
        return True
    
    def download_coco128_seg(self):
        """下载COCO128-seg数据集"""
        self.print_step(1, "下载COCO128-seg数据集")
        
        if self.coco128_seg_dir.exists():
            print("✅ COCO128-seg数据集已存在")
            return True
        
        zip_path = self.datasets_dir / "coco128-seg.zip"
        
        print("📥 正在下载COCO128-seg数据集...")
        print("下载地址: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip")
        
        try:
            response = requests.get('https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip')
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            print("✅ 下载完成")
            
            # 解压
            print("📦 正在解压数据集...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.datasets_dir)
            print("✅ 解压完成")
            
            # 清理zip文件
            zip_path.unlink()
            
            return True
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False
    
    def prepare_practice_dataset(self):
        """准备练习数据集"""
        self.print_step(2, "准备练习数据集")
        
        # 复制5张图像到练习目录
        source_images = list((self.coco128_seg_dir / "images" / "train2017").glob("*.jpg"))[:5]
        
        if not source_images:
            print("❌ 未找到源图像")
            return False
        
        print(f"📁 复制 {len(source_images)} 张图像到练习目录...")
        
        for img_path in source_images:
            shutil.copy2(img_path, self.practice_dir / img_path.name)
            print(f"  - 复制: {img_path.name}")
        
        # 创建类别文件
        classes_file = self.practice_dir / "classes.txt"
        classes = [
            "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light"
        ]
        
        with open(classes_file, 'w', encoding='utf-8') as f:
            for i, class_name in enumerate(classes):
                f.write(f"{i} {class_name}\n")
        
        print(f"✅ 创建类别文件: {classes_file}")
        
        # 创建标注指南
        guide_file = self.practice_dir / "实例分割标注指南.md"
        guide_content = """# 实例分割标注指南

## 🎯 标注目标
使用LabelMe为图像中的物体创建多边形标注，用于实例分割任务。

## 📋 标注步骤

### 1. 启动LabelMe
```bash
labelme datasets/segmentation_practice
```

### 2. 标注流程
1. **选择图像**: 在LabelMe中打开一张图像
2. **创建多边形**: 点击 "Create Polygon" 按钮
3. **绘制轮廓**: 沿着物体边界点击，创建多边形顶点
4. **闭合多边形**: 双击最后一个点或按Enter键闭合
5. **输入类别**: 在弹出的对话框中输入类别名称
6. **保存标注**: 按Ctrl+S保存当前图像的标注

### 3. 标注技巧
- **精确边界**: 尽量沿着物体的精确边界绘制
- **顶点密度**: 在曲线处增加更多顶点
- **类别一致**: 使用classes.txt中的标准类别名称
- **完整标注**: 确保所有目标物体都被标注

### 4. 保存格式
- 标注文件保存为JSON格式
- 文件名与图像文件名对应
- 每个物体包含多边形坐标和类别信息

## ⚠️ 注意事项
- 标注质量直接影响模型性能
- 保持耐心，精确标注
- 可以多次调整多边形形状
- 建议先标注简单物体，再标注复杂物体
"""
        
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"✅ 创建标注指南: {guide_file}")
        return True
    
    def show_dataset_info(self):
        """显示数据集信息"""
        self.print_step(3, "数据集信息")
        
        print("📊 COCO128-seg数据集:")
        if self.coco128_seg_dir.exists():
            image_count = len(list((self.coco128_seg_dir / "images" / "train2017").glob("*.jpg")))
            label_count = len(list((self.coco128_seg_dir / "labels" / "train2017").glob("*.txt")))
            print(f"  - 图像数量: {image_count}")
            print(f"  - 标签数量: {label_count}")
            print(f"  - 类别数量: 80个标准类别")
            print(f"  - 标签格式: 实例分割多边形坐标")
        else:
            print("  - 状态: 未下载")
        
        print(f"\n📁 练习数据集:")
        if self.practice_dir.exists():
            image_files = list(self.practice_dir.glob("*.jpg"))
            json_files = list(self.practice_dir.glob("*.json"))
            print(f"  - 图像数量: {len(image_files)}")
            print(f"  - 已标注数量: {len(json_files)}")
            print(f"  - 状态: {'✅ 准备就绪' if len(json_files) == 5 else '⏳ 需要标注'}")
        else:
            print("  - 状态: 未创建")
    
    def start_labelme_annotation(self):
        """启动LabelMe进行标注"""
        self.print_step(4, "启动LabelMe进行标注")
        
        if not self.practice_dir.exists():
            print("❌ 练习数据集未准备，请先运行步骤2")
            return False
        
        print("🚀 正在启动LabelMe...")
        print(f"标注目录: {self.practice_dir.absolute()}")
        
        print("\n📋 详细标注步骤:")
        print("1. LabelMe将自动打开练习目录")
        print("2. 选择一张图像开始标注")
        print("3. 点击 'Create Polygon' 创建多边形")
        print("4. 沿着物体边界点击，创建多边形顶点")
        print("5. 双击最后一个点或按Enter键闭合多边形")
        print("6. 输入类别名称 (参考classes.txt)")
        print("7. 按Ctrl+S保存标注")
        print("8. 重复步骤3-7，标注所有目标物体")
        print("9. 选择下一张图像继续标注")
        
        print(f"\n💡 重要提示:")
        print("- 使用多边形标注，不是矩形框")
        print("- 沿着物体的精确边界绘制")
        print("- 在曲线处增加更多顶点")
        print("- 确保多边形完全闭合")
        
        try:
            subprocess.run(["labelme", str(self.practice_dir)], check=True)
            print("\n✅ LabelMe已关闭")
            print("请检查标注文件是否已保存")
            
        except subprocess.CalledProcessError:
            print("❌ 启动LabelMe失败")
            print("请手动运行: labelme datasets/segmentation_practice")
        except FileNotFoundError:
            print("❌ 未找到LabelMe，请先安装: pip install labelme")
    
    def convert_to_yolo_seg(self):
        """转换为YOLO实例分割格式"""
        self.print_step(5, "转换为YOLO实例分割格式")
        
        # 检查是否有标注文件
        json_files = list(self.practice_dir.glob("*.json"))
        if not json_files:
            print("❌ 没有找到标注文件，请先完成标注")
            return False
        
        print(f"找到 {len(json_files)} 个标注文件")
        
        # 使用专门的转换脚本
        print("🔄 使用 labelme2yolo_seg.py 进行转换...")
        
        try:
            cmd = [
                sys.executable, "labelme2yolo_seg.py",
                "--input_dir", str(self.practice_dir),
                "--output_dir", str(self.output_dir),
                "--classes", str(self.practice_dir / "classes.txt")
            ]
            
            print(f"运行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ 转换完成!")
            print(result.stdout)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 转换失败: {e}")
            print(f"错误输出: {e.stderr}")
            return False
    
    def train_segmentation_model(self):
        """训练实例分割模型"""
        self.print_step(6, "训练实例分割模型")
        
        # 检查数据集
        dataset_yaml = self.output_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            print("❌ 数据集配置文件不存在，请先完成数据转换")
            return False
        
        print("🚀 开始训练实例分割模型...")
        print("注意: 实例分割训练需要更多时间和计算资源")
        
        # 训练命令
        cmd = [
            sys.executable, "segment/train.py",
            "--data", str(dataset_yaml),
            "--weights", "yolov5s-seg.pt",  # 使用实例分割预训练权重
            "--img", "640",
            "--epochs", "50",  # 实例分割需要更多epochs
            "--batch-size", "4",  # 减小批次大小
            "--project", "segmentation_training",
            "--name", "practice_model"
        ]
        
        print(f"训练命令: {' '.join(cmd)}")
        print("\n⏳ 训练开始，请耐心等待...")
        print("💡 提示:")
        print("- 实例分割训练比目标检测慢")
        print("- 会显示分割损失和边界框损失")
        print("- 训练结果保存在 runs/train/segmentation_training/")
        
        try:
            result = subprocess.run(cmd, check=True)
            print("\n🎉 训练完成!")
            
            # 显示结果位置
            result_dir = Path("runs/train/segmentation_training/practice_model")
            if result_dir.exists():
                print(f"\n📁 训练结果保存在: {result_dir}")
                
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 训练失败: {e}")
            print("请检查错误信息并重试")
            return False
        
        return True
    
    def validate_model(self):
        """验证模型性能"""
        self.print_step(7, "验证模型性能")
        
        # 查找最佳模型权重
        weights_dir = Path("runs/train/segmentation_training/practice_model/weights")
        if not weights_dir.exists():
            print("❌ 未找到训练结果，请先完成训练")
            return False
        
        best_weights = weights_dir / "best.pt"
        if not best_weights.exists():
            print("❌ 未找到最佳模型权重")
            return False
        
        print(f"✅ 找到最佳模型: {best_weights}")
        
        # 验证命令
        dataset_yaml = self.output_dir / "dataset.yaml"
        cmd = [
            sys.executable, "segment/val.py",
            "--weights", str(best_weights),
            "--data", str(dataset_yaml),
            "--img", "640"
        ]
        
        print(f"验证命令: {' '.join(cmd)}")
        print("\n⏳ 开始验证...")
        
        try:
            subprocess.run(cmd, check=True)
            print("\n✅ 验证完成!")
            print("查看验证结果了解模型性能")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 验证失败: {e}")
    
    def test_inference(self):
        """测试模型推理"""
        self.print_step(8, "测试模型推理")
        
        # 查找最佳模型权重
        weights_dir = Path("runs/train/segmentation_training/practice_model/weights")
        if not weights_dir.exists():
            print("❌ 未找到训练结果，请先完成训练")
            return False
        
        best_weights = weights_dir / "best.pt"
        if not best_weights.exists():
            print("❌ 未找到最佳模型权重")
            return False
        
        print(f"✅ 使用模型: {best_weights}")
        
        # 选择测试图像
        test_image = self.practice_dir / "000000000009.jpg"
        if not test_image.exists():
            print(f"❌ 测试图像不存在: {test_image}")
            return False
        
        # 推理命令
        cmd = [
            sys.executable, "segment/predict.py",
            "--weights", str(best_weights),
            "--source", str(test_image),
            "--project", "segmentation_inference",
            "--name", "practice_test"
        ]
        
        print(f"推理命令: {' '.join(cmd)}")
        print("\n⏳ 开始推理...")
        
        try:
            subprocess.run(cmd, check=True)
            print("\n✅ 推理完成!")
            
            # 显示结果位置
            result_dir = Path("runs/predict/segmentation_inference/practice_test")
            if result_dir.exists():
                print(f"\n📁 推理结果保存在: {result_dir}")
                print("查看生成的图像，对比原始标注和预测结果")
                
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 推理失败: {e}")
    
    def show_complete_workflow(self):
        """显示完整工作流程"""
        self.print_header("完整工作流程总结")
        
        workflow = """
🎯 实例分割完整学习流程:

1. 📥 数据集准备
   - 下载COCO128-seg官方数据集
   - 准备LabelMe练习数据集
   - 了解实例分割标签格式

2. 🏷️ 数据标注 (LabelMe)
   - 使用多边形标注工具
   - 沿着物体边界精确绘制
   - 标注5张练习图像

3. 🔄 格式转换
   - 将LabelMe JSON转换为YOLO格式
   - 分割训练集和验证集
   - 生成dataset.yaml配置文件

4. 🚀 模型训练
   - 使用yolov5s-seg.pt预训练权重
   - 训练50个epochs
   - 监控分割损失和边界框损失

5. 🔍 模型验证
   - 在验证集上评估性能
   - 查看mAP、分割精度等指标
   - 分析模型优缺点

6. 🎯 模型推理
   - 使用训练好的模型进行预测
   - 生成分割掩码和边界框
   - 对比预测结果和真实标注

💡 学习要点:
- 实例分割比目标检测更复杂
- 多边形标注需要更高的精度
- 训练时间更长，需要更多数据
- 结果包含分割掩码和检测框
"""
        
        print(workflow)
    
    def run_interactive_mode(self):
        """运行交互模式"""
        self.print_header("实例分割学习交互模式")
        
        while True:
            print("\n📋 请选择操作:")
            print("1. 🔍 检查环境")
            print("2. 📥 下载COCO128-seg数据集")
            print("3. 📁 准备练习数据集")
            print("4. 📊 显示数据集信息")
            print("5. 🏷️ 启动LabelMe进行标注")
            print("6. 🔄 转换为YOLO格式")
            print("7. 🚀 训练实例分割模型")
            print("8. 🔍 验证模型性能")
            print("9. 🎯 测试模型推理")
            print("10. 📋 查看完整工作流程")
            print("11. 🚪 退出")
            
            try:
                choice = input("\n请输入选择 (1-11): ").strip()
                
                if choice == '1':
                    self.check_environment()
                elif choice == '2':
                    self.download_coco128_seg()
                elif choice == '3':
                    self.prepare_practice_dataset()
                elif choice == '4':
                    self.show_dataset_info()
                elif choice == '5':
                    self.start_labelme_annotation()
                elif choice == '6':
                    self.convert_to_yolo_seg()
                elif choice == '7':
                    self.train_segmentation_model()
                elif choice == '8':
                    self.validate_model()
                elif choice == '9':
                    self.test_inference()
                elif choice == '10':
                    self.show_complete_workflow()
                elif choice == '11':
                    print("👋 再见！祝您学习愉快！")
                    break
                else:
                    print("❌ 无效选择，请输入1-11")
                    
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")


def main():
    """主函数"""
    print("🎯 YOLOv5 实例分割完整学习流程")
    print("=" * 60)
    
    learner = InstanceSegmentationLearning()
    
    # 检查环境
    if not learner.check_environment():
        return
    
    # 运行交互模式
    learner.run_interactive_mode()


if __name__ == "__main__":
    main() 