#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LabelMe JSON 转 YOLO 实例分割格式转换脚本

将LabelMe的多边形标注转换为YOLO实例分割所需的标签格式
实例分割标签包含：class_id x_center y_center width height polygon_points...

作者：AI助手
"""

import json
import os
import sys
from pathlib import Path
import argparse
from typing import List, Tuple, Dict


class LabelMe2YOLOSeg:
    def __init__(self, input_dir: str, output_dir: str, classes_file: str = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.classes_file = Path(classes_file) if classes_file else None
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 类别映射
        self.class_mapping = {}
        self.load_classes()
    
    def load_classes(self):
        """加载类别文件"""
        if self.classes_file and self.classes_file.exists():
            with open(self.classes_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            class_id = int(parts[0])
                            class_name = parts[1]
                            self.class_mapping[class_name] = class_id
                            print(f"加载类别: {class_id} -> {class_name}")
        else:
            # 默认类别
            default_classes = [
                "person", "bicycle", "car", "motorcycle", "airplane",
                "bus", "train", "truck", "boat", "traffic light"
            ]
            for i, class_name in enumerate(default_classes):
                self.class_mapping[class_name] = i
            print("使用默认类别映射")
    
    def add_new_class(self, class_name: str) -> int:
        """动态添加新类别"""
        if class_name not in self.class_mapping:
            new_id = len(self.class_mapping)
            self.class_mapping[class_name] = new_id
            print(f"➕ 添加新类别: {new_id} -> {class_name}")
        return self.class_mapping[class_name]
    
    def extract_classes_from_json(self, json_file: Path) -> List[str]:
        """从JSON文件中提取类别名称"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            classes = []
            for shape in data.get('shapes', []):
                label = shape.get('label', '').strip()
                if label and label not in classes:
                    classes.append(label)
            
            return classes
        except Exception as e:
            print(f"❌ 读取JSON文件失败 {json_file}: {e}")
            return []
    
    def convert_coordinates(self, points: List[List[float]], img_width: int, img_height: int) -> Tuple[List[float], List[float]]:
        """转换坐标：像素坐标 -> YOLO格式 (归一化)"""
        x_coords = []
        y_coords = []
        
        for point in points:
            x, y = point
            # 归一化到0-1范围
            x_norm = x / img_width
            y_norm = y / img_height
            
            # 确保在有效范围内
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            
            x_coords.append(x_norm)
            y_coords.append(y_norm)
        
        return x_coords, y_coords
    
    def calculate_bounding_box(self, x_coords: List[float], y_coords: List[float]) -> Tuple[float, float, float, float]:
        """计算边界框 (x_center, y_center, width, height)"""
        if not x_coords or not y_coords:
            return 0.0, 0.0, 0.0, 0.0
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        return x_center, y_center, width, height
    
    def convert_json_to_yolo_seg(self, json_file: Path) -> str:
        """将单个JSON文件转换为YOLO实例分割格式"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_width = data.get('imageWidth', 0)
            img_height = data.get('imageHeight', 0)
            
            if img_width == 0 or img_height == 0:
                print(f"⚠️  警告: {json_file.name} 缺少图像尺寸信息")
                return ""
            
            yolo_lines = []
            
            for shape in data.get('shapes', []):
                label = shape.get('label', '').strip()
                points = shape.get('points', [])
                shape_type = shape.get('shape_type', '')
                
                if not label or not points:
                    continue
                
                # 获取或添加类别ID
                class_id = self.add_new_class(label)
                
                # 转换坐标
                x_coords, y_coords = self.convert_coordinates(points, img_width, img_height)
                
                # 计算边界框
                x_center, y_center, width, height = self.calculate_bounding_box(x_coords, y_coords)
                
                # 构建YOLO实例分割标签行
                # 格式: class_id x_center y_center width height polygon_points...
                line_parts = [str(class_id), f"{x_center:.6f}", f"{y_center:.6f}", f"{width:.6f}", f"{height:.6f}"]
                
                # 添加多边形点坐标
                for x, y in zip(x_coords, y_coords):
                    line_parts.extend([f"{x:.6f}", f"{y:.6f}"])
                
                yolo_lines.append(" ".join(line_parts))
            
            return "\n".join(yolo_lines)
            
        except Exception as e:
            print(f"❌ 转换失败 {json_file}: {e}")
            return ""
    
    def convert(self):
        """执行转换"""
        print(f"🔄 开始转换 LabelMe -> YOLO 实例分割格式")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        
        # 查找所有JSON文件
        json_files = list(self.input_dir.glob("*.json"))
        if not json_files:
            print("❌ 未找到JSON标注文件")
            return False
        
        print(f"找到 {len(json_files)} 个JSON文件")
        
        # 创建训练和验证目录
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        # 分割数据集 (80%训练, 20%验证)
        total_files = len(json_files)
        train_count = int(total_files * 0.8)
        
        train_files = json_files[:train_count]
        val_files = json_files[train_count:]
        
        print(f"数据集分割: {len(train_files)} 训练, {len(val_files)} 验证")
        
        # 转换训练集
        print("\n📝 转换训练集...")
        for json_file in train_files:
            self.convert_single_file(json_file, train_dir)
        
        # 转换验证集
        print("\n📝 转换验证集...")
        for json_file in val_files:
            self.convert_single_file(json_file, val_dir)
        
        # 创建dataset.yaml文件
        self.create_dataset_yaml()
        
        print(f"\n✅ 转换完成!")
        print(f"输出目录: {self.output_dir}")
        print(f"训练集: {train_dir} ({len(train_files)} 个文件)")
        print(f"验证集: {val_dir} ({len(val_files)} 个文件)")
        
        return True
    
    def convert_single_file(self, json_file: Path, output_dir: Path):
        """转换单个文件"""
        # 转换标签
        yolo_content = self.convert_json_to_yolo_seg(json_file)
        if not yolo_content:
            print(f"⚠️  跳过 {json_file.name} (转换失败)")
            return
        
        # 生成输出文件名
        base_name = json_file.stem
        txt_file = output_dir / f"{base_name}.txt"
        
        # 保存YOLO格式标签
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(yolo_content)
        
        # 复制对应的图像文件
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in img_extensions:
            img_file = json_file.with_suffix(ext)
            if img_file.exists():
                import shutil
                shutil.copy2(img_file, output_dir / img_file.name)
                print(f"✅ {json_file.name} -> {txt_file.name} + {img_file.name}")
                break
        else:
            print(f"⚠️  未找到对应的图像文件: {json_file.name}")
    
    def create_dataset_yaml(self):
        """创建dataset.yaml配置文件"""
        yaml_content = f"""# 实例分割数据集配置
path: {self.output_dir.absolute()}
train: train
val: val

# 类别数量
nc: {len(self.class_mapping)}

# 类别名称
names:
"""
        
        # 按ID排序类别
        sorted_classes = sorted(self.class_mapping.items(), key=lambda x: x[1])
        for class_name, class_id in sorted_classes:
            yaml_content += f"  {class_id}: {class_name}\n"
        
        yaml_file = self.output_dir / "dataset.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"✅ 创建配置文件: {yaml_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LabelMe JSON 转 YOLO 实例分割格式")
    parser.add_argument("--input_dir", required=True, help="输入目录 (包含JSON文件)")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--classes", help="类别文件路径")
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not Path(args.input_dir).exists():
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    # 执行转换
    converter = LabelMe2YOLOSeg(args.input_dir, args.output_dir, args.classes)
    success = converter.convert()
    
    if success:
        print("\n🎉 转换成功完成!")
        print("\n💡 下一步:")
        print("1. 检查生成的标签文件")
        print("2. 使用 segment/train.py 训练实例分割模型")
        print("3. 确保使用 yolov5s-seg.pt 等实例分割预训练权重")
    else:
        print("\n❌ 转换失败")


if __name__ == "__main__":
    main() 