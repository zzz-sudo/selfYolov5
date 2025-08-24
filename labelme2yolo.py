#!/usr/bin/env python3
"""
LabelMe JSON格式转YOLO格式的转换脚本.

使用方法:
    python labelme2yolo.py --input_dir /path/to/labelme/json/files --output_dir /path/to/output --classes classes.txt

功能:
    1. 读取LabelMe标注的JSON文件
    2. 转换为YOLO格式的标签文件
    3. 自动分割训练集和验证集
    4. 生成数据集配置文件


"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import List

import yaml
from tqdm import tqdm


class LabelMe2YOLO:
    """LabelMe格式转YOLO格式的转换器."""

    def __init__(self, input_dir: str, output_dir: str, classes_file: str = None):
        """
        初始化转换器.

        Args:
            input_dir: LabelMe JSON文件所在目录
            output_dir: 输出目录
            classes_file: 类别文件路径
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.classes_file = Path(classes_file) if classes_file else None

        # 创建输出目录结构
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # 类别映射
        self.class_mapping = {}
        self.class_names = []

        # 统计信息
        self.stats = {"total_images": 0, "total_annotations": 0, "class_counts": {}, "conversion_errors": 0}

    def load_classes(self):
        """加载类别文件."""
        if self.classes_file and self.classes_file.exists():
            with open(self.classes_file, encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f.readlines() if line.strip()]
            print(f"加载了 {len(self.class_names)} 个类别: {self.class_names}")
        else:
            print("未找到类别文件，将从JSON文件中自动提取类别")

    def extract_classes_from_json(self, json_files: List[Path]):
        """从JSON文件中提取类别信息."""
        class_set = set()
        for json_file in tqdm(json_files, desc="提取类别信息"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                for shape in data.get("shapes", []):
                    label = shape.get("label", "").strip()
                    if label:
                        class_set.add(label)

            except Exception as e:
                print(f"读取文件 {json_file} 时出错: {e}")
                continue

        self.class_names = sorted(list(class_set))
        print(f"从JSON文件中提取了 {len(self.class_names)} 个类别: {self.class_names}")

        # 保存类别文件
        classes_file = self.output_dir / "classes.txt"
        with open(classes_file, "w", encoding="utf-8") as f:
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}\n")
        print(f"类别文件已保存到: {classes_file}")

    def convert_coordinates(self, points: List[List[float]], img_width: int, img_height: int) -> List[float]:
        """
        将LabelMe的坐标转换为YOLO格式.

        Args:
            points: LabelMe格式的点坐标 [[x1, y1], [x2, y2], ...]
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            YOLO格式的坐标 [x_center, y_center, width, height] (归一化)
        """
        if len(points) < 2:
            return []

        # 计算边界框
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # 转换为YOLO格式 (归一化坐标)
        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        # 确保坐标在[0, 1]范围内
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        return [x_center, y_center, width, height]

    def convert_json_to_yolo(self, json_file: Path) -> bool:
        """
        转换单个JSON文件为YOLO格式.

        Args:
            json_file: JSON文件路径

        Returns:
            转换是否成功
        """
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # 获取图像信息
            img_width = data.get("imageWidth", 0)
            img_height = data.get("imageHeight", 0)

            if img_width == 0 or img_height == 0:
                print(f"警告: {json_file} 中缺少图像尺寸信息")
                return False

            # 查找对应的图像文件
            img_filename = data.get("imagePath", "")
            if not img_filename:
                # 尝试从JSON文件名推断图像文件名
                img_filename = json_file.stem + ".jpg"  # 假设是jpg格式

            # 查找图像文件
            img_file = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                potential_img = self.input_dir / (json_file.stem + ext)
                if potential_img.exists():
                    img_file = potential_img
                    break

            if not img_file or not img_file.exists():
                print(f"警告: 未找到图像文件 {img_filename}")
                return False

            # 创建标签文件
            label_filename = json_file.stem + ".txt"
            label_file = self.labels_dir / label_filename

            annotations = []

            # 处理每个标注形状
            for shape in data.get("shapes", []):
                label = shape.get("label", "").strip()
                if not label:
                    continue

                # 获取类别ID
                if label in self.class_names:
                    class_id = self.class_names.index(label)
                else:
                    print(f"警告: 未知类别 '{label}' 在文件 {json_file}")
                    continue

                # 获取点坐标
                points = shape.get("points", [])
                if len(points) < 2:
                    continue

                # 转换坐标
                yolo_coords = self.convert_coordinates(points, img_width, img_height)
                if yolo_coords:
                    annotations.append(f"{class_id} {' '.join([f'{coord:.6f}' for coord in yolo_coords])}")

                    # 更新统计信息
                    self.stats["total_annotations"] += 1
                    if label not in self.stats["class_counts"]:
                        self.stats["class_counts"][label] = 0
                    self.stats["class_counts"][label] += 1

            # 写入标签文件
            if annotations:
                with open(label_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(annotations))

                # 复制图像文件
                img_dest = self.images_dir / img_file.name
                shutil.copy2(img_file, img_dest)

                return True

        except Exception as e:
            print(f"转换文件 {json_file} 时出错: {e}")
            self.stats["conversion_errors"] += 1
            return False

        return False

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.2):
        """
        分割数据集为训练集和验证集.

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        # 获取所有图像文件
        image_files = list(self.images_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]]

        if not image_files:
            print("未找到图像文件")
            return

        # 随机打乱
        random.shuffle(image_files)

        # 计算分割点
        total = len(image_files)
        train_end = int(total * train_ratio)

        train_files = image_files[:train_end]
        val_files = image_files[train_end:]

        # 创建训练集和验证集目录
        train_img_dir = self.output_dir / "train" / "images"
        train_label_dir = self.output_dir / "train" / "labels"
        val_img_dir = self.output_dir / "val" / "images"
        val_label_dir = self.output_dir / "val" / "labels"

        for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 移动训练集文件
        print("创建训练集...")
        for img_file in tqdm(train_files, desc="训练集"):
            # 移动图像
            shutil.copy2(img_file, train_img_dir / img_file.name)

            # 移动对应的标签文件
            label_file = self.labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy2(label_file, train_label_dir / label_file.name)

        # 移动验证集文件
        print("创建验证集...")
        for img_file in tqdm(val_files, desc="验证集"):
            # 移动图像
            shutil.copy2(img_file, val_img_dir / img_file.name)

            # 移动对应的标签文件
            label_file = self.labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy2(label_file, val_label_dir / label_file.name)

        print(f"数据集分割完成: 训练集 {len(train_files)} 张, 验证集 {len(val_files)} 张")

    def create_dataset_yaml(self):
        """创建数据集配置文件."""
        yaml_content = {
            "path": str(self.output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        yaml_file = self.output_dir / "dataset.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

        print(f"数据集配置文件已创建: {yaml_file}")

    def convert(self, train_ratio: float = 0.8, val_ratio: float = 0.2):
        """
        执行转换.

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        print("开始转换 LabelMe 格式到 YOLO 格式...")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")

        # 查找所有JSON文件
        json_files = list(self.input_dir.glob("*.json"))
        if not json_files:
            print("未找到JSON文件")
            return

        print(f"找到 {len(json_files)} 个JSON文件")

        # 加载或提取类别
        if self.class_names:
            self.load_classes()
        else:
            self.extract_classes_from_json(json_files)

        # 转换所有文件
        print("开始转换文件...")
        successful_conversions = 0

        for json_file in tqdm(json_files, desc="转换进度"):
            if self.convert_json_to_yolo(json_file):
                successful_conversions += 1

        self.stats["total_images"] = successful_conversions

        # 分割数据集
        print("分割数据集...")
        self.split_dataset(train_ratio, val_ratio)

        # 创建配置文件
        self.create_dataset_yaml()

        # 打印统计信息
        self.print_stats()

        print("转换完成!")

    def print_stats(self):
        """打印转换统计信息."""
        print("\n" + "=" * 50)
        print("转换统计信息")
        print("=" * 50)
        print(f"总图像数: {self.stats['total_images']}")
        print(f"总标注数: {self.stats['total_annotations']}")
        print(f"类别数量: {len(self.class_names)}")
        print(f"转换错误: {self.stats['conversion_errors']}")

        if self.stats["class_counts"]:
            print("\n各类别标注数量:")
            for class_name, count in sorted(self.stats["class_counts"].items()):
                print(f"  {class_name}: {count}")

        print("=" * 50)


def main():
    """主函数."""
    parser = argparse.ArgumentParser(description="LabelMe格式转YOLO格式")
    parser.add_argument("--input_dir", required=True, help="LabelMe JSON文件所在目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--classes", help="类别文件路径 (可选)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例 (默认: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例 (默认: 0.2)")

    args = parser.parse_args()

    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return

    # 创建转换器
    converter = LabelMe2YOLO(args.input_dir, args.output_dir, args.classes)

    # 执行转换
    converter.convert(args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()
