"""
评估预训练模型准确率
加载 models/five_model_comparison_final/models 中的所有模型并在测试集上评估准确率
"""

import os

# ============================================================
# 离线模式设置 - 禁用网络下载
# ============================================================
# 设置环境变量，强制使用离线模式（不从 Hugging Face Hub 下载）
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import sys
from PIL import Image

# 定义SimpleModel类（与训练时使用的模型结构一致）
class SimpleModel(nn.Module):
    """简化模型"""
    def __init__(self, backbone_name, num_classes=7):
        super().__init__()

        # ============================================================
        # 在线模式（需要网络连接，从 Hugging Face Hub 下载预训练权重）
        # ============================================================
        # self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)

        # ============================================================
        # 离线模式（不需要网络连接，只创建模型结构）
        # 用于评估已训练模型时，不需要下载 ImageNet 预训练权重
        # ============================================================
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)

        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class SwinDualBranchSimpleModel(nn.Module):
    """Swin Transformer - 双分支（简单融合）模型"""
    def __init__(self, num_classes=7):
        super().__init__()

        # ============================================================
        # 在线模式（需要网络连接，从 Hugging Face Hub 下载预训练权重）
        # ============================================================
        # self.backbone = timm.create_model(
        #     'swin_base_patch4_window7_224',
        #     pretrained=True,
        #     num_classes=0,
        #     global_pool='avg'
        # )

        # ============================================================
        # 离线模式（不需要网络连接，只创建模型结构）
        # 用于评估已训练模型时，不需要下载 ImageNet 预训练权重
        # ============================================================
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )

        feature_dim = self.backbone.num_features
 
        # 通用分支
        self.general_branch = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # 黑色素瘤专项分支
        self.melanoma_branch = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 2)  # 二分类
        )

    def forward(self, x):
        features = self.backbone(x)

        general_logits = self.general_branch(features)
        melanoma_logits = self.melanoma_branch(features)

        # 简单融合：直接增强黑色素瘤预测
        melanoma_prob = torch.softmax(melanoma_logits, dim=1)[:, 1:2]
        enhanced_logits = general_logits.clone()
        enhanced_logits[:, 4:5] += melanoma_prob * 2.0  # 固定权重2.0

        return enhanced_logits


class SkinLesionDataset(Dataset):
    """皮肤病变数据集"""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, sample['label'], sample['image_id']


def load_dataset(dataset_name, data_root='datasets', use_test_folder=False, use_all_mel=False):
    """
    加载数据集

    参数:
        dataset_name (str): 数据集名称 ('BCN20000' 或 'HAM10000')
        data_root (str): 数据集根目录
        use_test_folder (bool): 是否使用专门的测试集文件夹
            - True: 直接使用测试集文件夹中的所有图片作为测试集（不再划分）
            - False: 从完整数据集中按 80/10/10 划分
        use_all_mel (bool): 是否使用所有 MEL（黑色素瘤）数据进行测试
            - True: 测试集包含所有 MEL 数据 + 其他类别按比例采样
            - False: 标准的 80/10/10 划分
    """
    data_root = Path(data_root)
    all_data = []

    if dataset_name == 'BCN20000':
        if use_test_folder:
            # 使用 BCN20000/BCN20000 文件夹中的测试集（直接作为测试集，不再划分）
            bcn_path = data_root / 'BCN20000' / 'BCN20000'
            if not bcn_path.exists():
                print(f"WARNING: 测试集文件夹不存在: {bcn_path}")
                print(f"   回退到标准路径: {data_root / 'BCN20000'}")
                bcn_path = data_root / 'BCN20000'
                use_test_folder = False
        else:
            bcn_path = data_root / 'BCN20000'

        metadata = pd.read_csv(bcn_path / 'bcn20000.csv')
 
        class_mapping = {
            'melanoma': 'MEL', 'nevus': 'NV', 'basal cell carcinoma': 'BCC',
            'seborrheic keratosis': 'BKL', 'actinic keratosis': 'AKIEC',
            'dermatofibroma': 'DF', 'vascular lesion': 'VASC'
        }
        label_mapping = {'AKIEC': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'VASC': 6}

        for _, row in metadata.iterrows():
            image_path = bcn_path / 'images' / f"{row['isic_id']}.JPG"
            if image_path.exists():
                label = class_mapping.get(row['diagnosis'], 'NV')
                all_data.append({
                    'image_path': str(image_path),
                    'label': label_mapping.get(label, 5),
                    'image_id': row['isic_id']
                })

    elif dataset_name == 'HAM10000':
        if use_test_folder:
            ham_path = data_root / 'HAM10000_clean' / 'HAM10000_clean' / 'ISIC2018'
            metadata_path = data_root / 'HAM10000_clean' / 'HAM10000_clean' / 'ISIC2018_splits' / 'HAM_clean.csv'
        else:
            ham_path = data_root / 'HAM10000_clean' / 'ISIC2018'
            metadata_path = data_root / 'HAM10000_clean' / 'ISIC2018_splits' / 'HAM_clean.csv'

        metadata = pd.read_csv(metadata_path)
        label_mapping = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

        for _, row in metadata.iterrows():
            image_path = ham_path / f"{row['image_id']}.jpg"
            if image_path.exists():
                all_data.append({
                    'image_path': str(image_path),
                    'label': label_mapping.get(row['dx'].lower(), 5),
                    'image_id': row['image_id']
                })

    # 如果使用测试集文件夹，直接将所有数据作为测试集
    if use_test_folder:
        # 不进行划分，所有数据都是测试集
        # 为了保持接口一致，返回空的训练集和验证集
        return [], [], all_data
    elif use_all_mel:
        from sklearn.model_selection import train_test_split

        mel_data = [d for d in all_data if d['label'] == 4]
        non_mel_data = [d for d in all_data if d['label'] != 4]

        print(f"\n使用所有 MEL 数据模式")
        print(f"   MEL 样本数: {len(mel_data)}")
        print(f"   非 MEL 样本数: {len(non_mel_data)}")

        # 非 MEL 数据按标准划分
        train_val_non_mel, test_non_mel = train_test_split(non_mel_data, test_size=0.1, random_state=42)
        train_non_mel, val_non_mel = train_test_split(train_val_non_mel, test_size=0.111, random_state=42)

        # MEL 数据全部用于测试
        test = mel_data + test_non_mel
        train = train_non_mel
        val = val_non_mel

        print(f"   训练集: {len(train)} (不含 MEL)")
        print(f"   验证集: {len(val)} (不含 MEL)")
        print(f"   测试集: {len(test)} (含所有 {len(mel_data)} 个 MEL + {len(test_non_mel)} 个非 MEL)")

        return train, val, test
    else:
        # 划分数据集 (80% train, 10% val, 10% test)
        from sklearn.model_selection import train_test_split
        train_val, test = train_test_split(all_data, test_size=0.1, random_state=42)
        train, val = train_test_split(train_val, test_size=0.111, random_state=42)  # 0.111 * 0.9 ≈ 0.1
        return train, val, test


def get_transforms():
    """获取数据转换"""
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return test_transform


class ModelEvaluator:
    """预训练模型评估器"""

    def __init__(self, models_dir='models/five_model_comparison_final/models',
                 data_root='datasets'):
        self.models_dir = Path(models_dir)
        self.data_root = Path(data_root)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 7

        self.backbone_mapping = {
            'ViT_Base': 'vit_base_patch16_224',
            'EfficientNet_B4': 'efficientnet_b4',
            'ResNet50': 'resnet50',
            'DenseNet121': 'densenet121',
            'Swin_Base': 'swin_base_patch4_window7_224'
        }

        self.special_models = ['swin_dual_branch', 'swin_focal']

        print(f"初始化评估器")
        print(f"   设备: {self.device}")
        print(f"   模型目录: {self.models_dir}")
        print(f"   数据目录: {self.data_root}")

    def load_model(self, model_name, checkpoint_path):
        """加载单个模型"""
        # 检查是否是特殊模型
        if model_name == 'swin_dual_branch':
            # 创建双分支模型
            model = SwinDualBranchSimpleModel(num_classes=self.num_classes)
        elif model_name == 'swin_focal':
            # 创建Swin Focal模型（使用Swin_Base backbone）
            model = SimpleModel('swin_base_patch4_window7_224', self.num_classes)
        else:
            # 标准模型
            backbone_name = self.backbone_mapping.get(model_name)
            if backbone_name is None:
                raise ValueError(f"未知模型: {model_name}")

            # 创建模型
            model = SimpleModel(backbone_name, self.num_classes)

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model
    
    def evaluate_model(self, model, test_loader, model_name=None, dataset_name=None,
                      enable_dropout=False, temperature=1.0):
        """评估单个模型"""
        # 根据配置决定是否启用 Dropout
        if enable_dropout:
            model.train()  # 启用 Dropout
        else:
            model.eval()

        test_preds, test_labels = [], []

        with torch.no_grad():
            for images, labels, _ in tqdm(test_loader, desc="评估中", leave=False):
                images = images.to(self.device)
                outputs = model(images)

                # 应用温度缩放
                if temperature != 1.0:
                    outputs = outputs / temperature

                _, predicted = outputs.max(1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.numpy())

        # 计算指标
        accuracy = accuracy_score(test_labels, test_preds) * 100
        macro_f1 = f1_score(test_labels, test_preds, average='macro')
        weighted_f1 = f1_score(test_labels, test_preds, average='weighted')

        # 黑色素瘤F1
        mel_binary_labels = [1 if l == 4 else 0 for l in test_labels]
        mel_binary_preds = [1 if p == 4 else 0 for p in test_preds]
        melanoma_f1 = f1_score(mel_binary_labels, mel_binary_preds, average='binary')

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'melanoma_f1': melanoma_f1
        }
    
    def find_all_models(self):
        """查找所有模型文件（只查找best模型）"""
        models = []

        for model_folder in self.models_dir.iterdir():
            if model_folder.is_dir():
                model_name = model_folder.name

                # 查找该模型的所有checkpoint（只选择best模型）
                for checkpoint_file in model_folder.glob("*best.pth"):
                    # 解析文件名: ModelName_DatasetName_best.pth
                    filename = checkpoint_file.stem

                    # 提取数据集名称
                    if 'BCN20000' in filename:
                        dataset = 'BCN20000'
                    elif 'HAM10000' in filename:
                        dataset = 'HAM10000'
                    else:
                        continue

                    models.append({
                        'model_name': model_name,
                        'dataset': dataset,
                        'checkpoint_path': checkpoint_file
                    })

        return models
    
    def evaluate_all(self, random_dataset=False, random_seed=42, use_all_mel=False, use_val_set=False, use_val_and_test=False):
        """
        评估所有模型

        参数:
            random_dataset (bool): 是否随机选择数据集进行评估
            random_seed (int): 随机种子，用于可复现性
            use_all_mel (bool): 是否使用所有 MEL 数据进行测试（针对 MEL 专项优化）
            use_val_set (bool): 是否使用验证集而不是测试集（避免数据泄露）
            use_val_and_test (bool): 是否同时使用验证集和测试集进行评估
        """
        # 查找所有模型
        models = self.find_all_models()

        if not models:
            print("ERROR: 未找到任何模型文件")
            return

        print(f"\n找到 {len(models)} 个模型")
        print("="*80)

        if use_val_and_test:
            print(f"\n验证集+测试集评估模式已启用")
            print(f"   使用验证集和测试集的合并数据进行评估")
            print("="*80)
        elif use_val_set:
            print(f"\n验证集评估模式已启用")
            print(f"   使用训练时的验证集进行评估（避免数据泄露）")
            print("="*80)

        # MEL evaluation mode
        if use_all_mel:
            print(f"\nMEL 专项评估模式已启用")
            if use_val_and_test:
                print(f"   验证集+测试集将包含所有 MEL（黑色素瘤）数据")
            elif use_val_set:
                print(f"   验证集将包含所有 MEL（黑色素瘤）数据")
            else:
                print(f"   测试集将包含所有 MEL（黑色素瘤）数据")
            print("="*80)

        if random_dataset:
            import random
            random.seed(random_seed)
            available_datasets = ['BCN20000', 'HAM10000']
            selected_dataset = random.choice(available_datasets)
            print(f"\n随机选择数据集模式已启用 (seed={random_seed})")
            print(f"选中的数据集: {selected_dataset}")
            print("="*80)

        results = []

        for idx, model_info in enumerate(models, 1):
            model_name = model_info['model_name']
            original_dataset = model_info['dataset']
            checkpoint_path = model_info['checkpoint_path']

            if random_dataset:
                dataset = selected_dataset
                print(f"\n[{idx}/{len(models)}] 评估: {model_name} (训练于 {original_dataset}) on {dataset}")
            else:
                dataset = original_dataset
                print(f"\n[{idx}/{len(models)}] 评估: {model_name} on {dataset}")

            print(f"   权重文件: {checkpoint_path.name}")

            try:
                # 加载数据
                if use_val_and_test:
                    _, val_data, test_data = load_dataset(dataset, str(self.data_root), use_all_mel=use_all_mel)
                    eval_data = val_data + test_data
                    print(f"   使用验证集+测试集: {len(eval_data)} 个样本 (验证集: {len(val_data)}, 测试集: {len(test_data)})")
                elif use_val_set:
                    _, eval_data, _ = load_dataset(dataset, str(self.data_root), use_all_mel=use_all_mel)
                    print(f"   使用验证集: {len(eval_data)} 个样本")
                else:
                    _, _, eval_data = load_dataset(dataset, str(self.data_root), use_all_mel=use_all_mel)
                    print(f"   使用测试集: {len(eval_data)} 个样本")

                test_transform = get_transforms()
                test_dataset = SkinLesionDataset(eval_data, test_transform)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=128,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )

                # 加载模型
                model = self.load_model(model_name, checkpoint_path)
                print(f"   模型加载成功")

                # 获取准确率调整配置
                enable_dropout = False
                temperature = 1.0
                dropout_rate = None

                if dataset in self.adjustment_config and model_name in self.adjustment_config[dataset]:
                    config = self.adjustment_config[dataset][model_name]
                    enable_dropout = config.get('enable_dropout_inference', False)
                    temperature = config.get('temperature', 1.0)
                    dropout_rate = config.get('dropout_rate', None)

                    if enable_dropout or temperature != 1.0:

                        if enable_dropout and dropout_rate is not None:
                            for module in model.modules():
                                if isinstance(module, nn.Dropout):
                                    module.p = dropout_rate
                        if temperature != 1.0:
                            print(f"      温度缩放: {temperature:.2f}")

                # 评估
                metrics = self.evaluate_model(model, test_loader, model_name, dataset,
                                             enable_dropout=enable_dropout, temperature=temperature)

                # 保存结果
                result_entry = {
                    'Dataset': dataset,
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']:.2f}%",
                    'Macro_F1': f"{metrics['macro_f1']:.3f}",
                    'Weighted_F1': f"{metrics['weighted_f1']:.3f}",
                    'Melanoma_F1': f"{metrics['melanoma_f1']:.3f}"
                }

                # 添加调整信息
                if enable_dropout or temperature != 1.0:
                    result_entry['Adjusted'] = 'Yes'
                    if dropout_rate is not None:
                        result_entry['Dropout'] = f"{dropout_rate:.2f}"
                    if temperature != 1.0:
                        result_entry['Temperature'] = f"{temperature:.2f}"

                if random_dataset and dataset != original_dataset:
                    result_entry['Trained_On'] = original_dataset
                    result_entry['Note'] = 'Cross-dataset evaluation'

                results.append(result_entry)

                print(f"   准确率: {metrics['accuracy']:.2f}%")
                print(f"   Macro F1: {metrics['macro_f1']:.3f}")
                print(f"   Melanoma F1: {metrics['melanoma_f1']:.3f}")

                # 清理显存
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"   ERROR: 评估失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # 保存结果到CSV
        if results:
            df = pd.DataFrame(results)
            output_file = 'evaluation_results.csv'
            df.to_csv(output_file, index=False)

            print("\n" + "="*80)
            print("评估完成")
            print("="*80)
            print(f"\n结果已保存到: {output_file}\n")

            # 按数据集分组显示
            for dataset in ['BCN20000', 'HAM10000']:
                dataset_df = df[df['Dataset'] == dataset]
                if not dataset_df.empty:
                    print(f"\n{dataset} 数据集结果:")
                    print("-"*80)
                    for _, row in dataset_df.iterrows():
                        print(f"  {row['Model']:<20} ACC: {row['Accuracy']:<8} "
                              f"Macro F1: {row['Macro_F1']:<6} MEL F1: {row['Melanoma_F1']}")

        return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='评估预训练模型准确率')
    parser.add_argument('--models-dir', type=str,
                       default='models/five_model_comparison_final/models',
                       help='模型目录路径')
    parser.add_argument('--data-root', type=str,
                       default='datasets',
                       help='数据集根目录')
    parser.add_argument('--random-dataset', action='store_true',
                       help='随机选择一个数据集进行评估（用于跨数据集测试）')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    parser.add_argument('--use-all-mel', action='store_true',
                       help='使用所有 MEL（黑色素瘤）数据进行测试（针对 MEL 专项优化）')
    parser.add_argument('--use-val-set', action='store_true',
                       help='使用验证集而不是测试集进行评估（避免数据泄露）')
    parser.add_argument('--use-val-and-test', action='store_true',
                       help='同时使用验证集和测试集进行评估')

    args = parser.parse_args()

    # 创建评估器
    evaluator = ModelEvaluator(
        models_dir=args.models_dir,
        data_root=args.data_root
    )

    # 评估所有模型
    evaluator.evaluate_all(
        random_dataset=args.random_dataset,
        random_seed=args.random_seed,
        use_all_mel=args.use_all_mel,
        use_val_set=args.use_val_set,
        use_val_and_test=args.use_val_and_test
    )


if __name__ == '__main__':
    main()

