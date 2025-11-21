"""
SwinTransformer消融实验
分离4个创新点，逐步验证每个创新点的贡献

创新点:
1. Focal Loss (vs CrossEntropy)
2. Swin架构 (vs ViT)
3. 双分支结构 (vs 单分支)
4. 注意力融合 (vs 简单融合)

实验设计（从弱到强）:
- Baseline: ViT + CrossEntropy + 单分支
- Exp1: ViT + Focal Loss + 单分支
- Exp2: Swin + Focal Loss + 单分支
- Exp3: Swin + Focal Loss + 双分支（简单融合）
- Exp4: Swin + Focal Loss + 双分支 + 注意力融合 (完整方案)
"""

import os
import sys

# ============================================================
# 离线模式设置 - 禁用网络下载
# ============================================================
# 设置环境变量，强制使用离线模式（不从 Hugging Face Hub 下载）
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 疾病名称
DISEASE_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

class MelanomaDataset(Dataset):
    """皮肤病数据集 - 完全模仿five_model_comparison"""
    def __init__(self, samples, transform=None, is_training=True):
        self.samples = samples
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        label = sample['label']

        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # 如果加载失败，创建黑色图像
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label

def create_melanoma_focused_transforms():
    """创建数据增强 - 完全模仿five_model_comparison"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

class FocalLoss(nn.Module):
    """Focal Loss实现"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BaselineViTModel(nn.Module):
    """基线ViT模型 - 单分支"""
    def __init__(self, num_classes=7):
        super().__init__()

        # ============================================================
        # 在线模式（需要网络连接，从 Hugging Face Hub 下载预训练权重）
        # ============================================================
        # self.backbone = timm.create_model(
        #     'vit_base_patch16_224',
        #     pretrained=True,
        #     num_classes=0,
        #     global_pool='avg'
        # )

        # ============================================================
        # 离线模式（不需要网络连接，只创建模型结构）
        # ============================================================
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.num_features, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class SwinSingleBranchModel(nn.Module):
    """Swin Transformer - 单分支"""
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
        # ============================================================
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.num_features, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class SwinDualBranchSimpleModel(nn.Module):
    """Swin Transformer - 双分支（简单融合）"""
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
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
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

class SwinDualBranchAttentionModel(nn.Module):
    """Swin Transformer - 双分支 + 注意力融合（完整方案）"""
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
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        # 黑色素瘤专项分支
        self.melanoma_branch = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )
        
        # 注意力融合模块
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        general_logits = self.general_branch(features)
        melanoma_logits = self.melanoma_branch(features)
        
        # 注意力权重
        attention_weights = self.attention(features)
        
        # 注意力融合
        melanoma_prob = torch.softmax(melanoma_logits, dim=1)[:, 1:2]
        enhanced_logits = general_logits.clone()
        
        # 使用注意力权重动态调整增强强度
        enhancement_strength = attention_weights[:, 1:2] * 3.0  # 动态权重
        enhanced_logits[:, 4:5] += melanoma_prob * enhancement_strength
        
        return enhanced_logits

class AblationConfig:
    """消融实验配置 - 完全模仿five_model_comparison_separate_datasets.py"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 训练配置（与five_model_comparison保持一致）
        self.epochs = 50
        self.batch_size = 64  # 降低batch_size避免显存溢出
        self.num_workers = 4
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.patience = 999
        self.num_classes = 7

        # 硬件优化配置
        self.pin_memory = True
        self.prefetch_factor = 2
        self.persistent_workers = False
        self.gradient_accumulation_steps = 1  # 梯度累积，等效batch_size=256

        # 输出配置
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"swin_ablation_{timestamp}"
        self.results_dir = f"results/{self.session_name}"
        self.models_dir = f"models/{self.session_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # 日志文件
        self.log_file = os.path.join(self.results_dir, 'training_log.txt')
        
        # 完整实验序列（4步）
        # Step 1: Baseline: ViT + WeightedCE = 87.12% (BCN20000) / 88.42% (HAM10000) 
        # Step 2: ViT + Focal Loss 
        # Step 3: Swin + Focal Loss 
        # Step 4: Swin + Focal + 双分支 
        self.experiments = {
            # Step 2已完成，注释掉
            # 'vit_focal': {
            #     'name': 'Step 2: ViT + Focal Loss',
            #     'model_class': BaselineViTModel,
            #     'use_focal_loss': True,
            #     'priority': 3,
            #     'expected_acc': 89.0
            # },
            # Step 3跳过（显存不足）
            'swin_focal': {
                'name': 'Step 3: Swin + Focal Loss',
                'model_class': SwinSingleBranchModel,
                'use_focal_loss': True,
                'priority': 2,
                'expected_acc': 91.0
            },
            'swin_dual_simple': {
                'name': 'Step 4: Swin + Focal + 双分支',
                'model_class': SwinDualBranchSimpleModel,
                'use_focal_loss': True,
                'priority': 1,
                'expected_acc': 93.0  # 基线87% + 所有创新点 6%
            }
        }

        print(f"SwinTransformer消融实验配置")
        print("=" * 60)
        print(f"实验数量: {len(self.experiments)}")
        print(f"训练轮数: {self.epochs}")
        print(f"批次大小: {self.batch_size}")
        print(f"结果保存: {self.results_dir}")
        print("=" * 60)

class AblationTrainer:
    """消融实验训练器"""
    def __init__(self, model, config, device, criterion, exp_name='', dataset_name=''):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = criterion
        self.exp_name = exp_name
        self.dataset_name = dataset_name

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )

        self.best_acc = 0.0
        self.patience_counter = 0

    def log_message(self, message):
        """记录日志到文件和控制台"""
        print(message)
        with open(self.config.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def save_model(self, epoch, acc, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': acc,
            'exp_name': self.exp_name,
            'dataset_name': self.dataset_name
        }

        # 保存最新模型
        latest_path = os.path.join(
            self.config.models_dir,
            f'{self.dataset_name}_{self.exp_name}_latest.pth'
        )
        torch.save(checkpoint, latest_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(
                self.config.models_dir,
                f'{self.dataset_name}_{self.exp_name}_best.pth'
            )
            torch.save(checkpoint, best_path)
            self.log_message(f"保存最佳模型: {best_path}")
            self.log_message(f"   准确率: {acc:.2f}%")
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()  # 在epoch开始时清零

        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 梯度累积
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            # 每accumulation_steps步更新一次
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        # 处理最后不足accumulation_steps的batch
        if (batch_idx + 1) % self.config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / len(train_loader), 100. * correct / total
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(self.device)
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        # 黑色素瘤F1
        mel_f1 = f1_score(
            [1 if l == 4 else 0 for l in all_labels],
            [1 if p == 4 else 0 for p in all_preds],
            average='binary'
        )
        
        return accuracy, f1_macro, f1_weighted, mel_f1
    
    def train(self, train_loader, val_loader):
        self.log_message(f"\n开始训练...")
        self.log_message(f"   实验: {self.exp_name}")
        self.log_message(f"   数据集: {self.dataset_name}")

        best_f1_macro = 0
        best_f1_weighted = 0
        best_mel_f1 = 0

        for epoch in range(self.config.epochs):
            msg = f"\nEpoch {epoch+1}/{self.config.epochs}"
            self.log_message(msg)

            train_loss, train_acc = self.train_epoch(train_loader)
            val_acc, f1_macro, f1_weighted, mel_f1 = self.validate(val_loader)

            self.scheduler.step()

            msg = f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
            self.log_message(msg)
            msg = f"Val Acc: {val_acc:.2f}%, F1 Macro: {f1_macro:.4f}, MEL F1: {mel_f1:.4f}"
            self.log_message(msg)

            # 保存模型和早停检查
            is_best = False
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                best_f1_macro = f1_macro
                best_f1_weighted = f1_weighted
                best_mel_f1 = mel_f1
                self.patience_counter = 0
                is_best = True
                self.log_message(f"新的最佳准确率: {val_acc:.2f}%")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    self.log_message(f"早停触发 (patience={self.config.patience})")
                    break

            # 保存模型
            self.save_model(epoch + 1, val_acc, is_best=is_best)

        self.log_message(f"\n训练完成!")
        self.log_message(f"   最佳准确率: {self.best_acc:.2f}%")
        self.log_message(f"   F1 Macro: {best_f1_macro:.4f}")
        self.log_message(f"   MEL F1: {best_mel_f1:.4f}")

        return self.best_acc, best_f1_macro, best_f1_weighted, best_mel_f1

def load_previous_results(dataset_name):
    """加载之前完成的实验结果（从five_model_comparison提取真实基线）"""
    previous_results = []

    if dataset_name == 'BCN20000':
        # 从five_model_comparison_separate_datasets提取的真实BasicViT结果
        # 来源: comprehensive_comparison_results.csv
        previous_results = [
            {
                'experiment': 'baseline_vit_weighted_ce',
                'name': 'Baseline: ViT + WeightedCE',
                'dataset': 'BCN20000',
                'best_accuracy': 87.12,  # 真实结果
                'f1_macro': 0.857,
                'f1_weighted': 0.871,
                'melanoma_f1': 0.877,
                'expected_acc': 87.0,
                'improvement': 0.0
            }
        ]
    elif dataset_name == 'HAM10000':
        # 从five_model_comparison_separate_datasets提取的真实BasicViT结果
        previous_results = [
            {
                'experiment': 'baseline_vit_weighted_ce',
                'name': 'Baseline: ViT + WeightedCE',
                'dataset': 'HAM10000',
                'best_accuracy': 88.42,  # 真实结果
                'f1_macro': 0.842,
                'f1_weighted': 0.884,
                'melanoma_f1': 0.754,
                'expected_acc': 88.0,
                'improvement': 0.0
            }
        ]

    return previous_results

def run_single_experiment(exp_name, exp_config, config, train_loader, val_loader, dataset_name, baseline_acc):
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"实验: {exp_config['name']}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")

    # 创建模型
    model = exp_config['model_class']()

    # 选择损失函数
    if exp_config['use_focal_loss']:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("使用Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用CrossEntropy Loss")

    # 创建训练器（传入exp_name和dataset_name用于保存模型）
    trainer = AblationTrainer(
        model, config, config.device, criterion,
        exp_name=exp_name,
        dataset_name=dataset_name
    )

    # 训练
    best_acc, f1_macro, f1_weighted, mel_f1 = trainer.train(train_loader, val_loader)

    results = {
        'experiment': exp_name,
        'name': exp_config['name'],
        'dataset': dataset_name,
        'best_accuracy': best_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'melanoma_f1': mel_f1,
        'expected_acc': exp_config['expected_acc'],
        'improvement': best_acc - baseline_acc  # vs真实baseline
    }

    print(f"\n实验完成!")
    print(f"最佳准确率: {best_acc:.2f}%")
    print(f"相对基线提升: +{results['improvement']:.2f}%")

    return results

def load_dataset(dataset_name):
    """加载指定数据集 - 完全模仿five_model_comparison_separate_datasets.py"""
    print(f"加载 {dataset_name} 数据集...")

    train_data = []
    val_data = []
    test_data = []

    if dataset_name == 'BCN20000':
        # 加载BCN20000数据集
        bcn_path = 'datasets/BCN20000'
        bcn_images_path = os.path.join(bcn_path, 'images')
        bcn_metadata_path = os.path.join(bcn_path, 'bcn20000.csv')

        if os.path.exists(bcn_images_path) and os.path.exists(bcn_metadata_path):
            print(f"找到BCN20000数据集: {bcn_images_path}")

            # 读取元数据
            metadata = pd.read_csv(bcn_metadata_path)
            print(f"   元数据记录: {len(metadata)}")
            print(f"   列名: {metadata.columns.tolist()}")

            # 类别映射
            class_mapping = {
                'melanoma': 'MEL', 'nevus': 'NV', 'basal cell carcinoma': 'BCC',
                'seborrheic keratosis': 'BKL', 'actinic keratosis': 'AKIEC',
                'squamous cell carcinoma': 'AKIEC', 'dermatofibroma': 'DF',
                'solar lentigo': 'BKL', 'vascular lesion': 'VASC'
            }

            # 处理数据
            for _, row in metadata.iterrows():
                image_id = row['isic_id']
                diagnosis = row['diagnosis']
                split = row['split'] if 'split' in row else 'train'

                # 映射到标准类别
                standard_label = class_mapping.get(diagnosis, 'NV')

                # 检查图像文件是否存在
                image_path = os.path.join(bcn_images_path, f"{image_id}.JPG")
                if os.path.exists(image_path):
                    # 转换为数字标签
                    label_mapping = {'AKIEC': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'VASC': 6}
                    numeric_label = label_mapping.get(standard_label, 5)

                    sample = {
                        'image_path': image_path,
                        'label': numeric_label,
                        'image_id': image_id
                    }

                    # 根据split分配
                    if split == 'train':
                        train_data.append(sample)
                    elif split == 'valid' or split == 'val':
                        val_data.append(sample)
                    elif split == 'test':
                        test_data.append(sample)

            print(f"   BCN20000成功加载:")
            print(f"      训练集: {len(train_data)} 样本")
            print(f"      验证集: {len(val_data)} 样本")
            print(f"      测试集: {len(test_data)} 样本")
        else:
            print(f"   BCN20000数据集未找到")
            print(f"      检查路径: {bcn_images_path}")
            print(f"      检查CSV: {bcn_metadata_path}")

    elif dataset_name == 'HAM10000':
        # 加载HAM10000数据集
        ham_paths_to_try = [
            'datasets/HAM10000_clean/ISIC2018',
            'datasets/HAM10000_clean/images'
        ]

        ham_metadata_paths_to_try = [
            'datasets/HAM10000_clean/ISIC2018_splits/HAM_clean.csv',
            'datasets/HAM10000_clean/HAM10000_metadata.csv'
        ]

        ham_loaded = False
        for ham_img_path, ham_meta_path in zip(ham_paths_to_try, ham_metadata_paths_to_try):
            if os.path.exists(ham_img_path) and os.path.exists(ham_meta_path):
                print(f"找到HAM10000数据集: {ham_img_path}")

                try:
                    # 读取HAM10000元数据
                    ham_metadata = pd.read_csv(ham_meta_path)
                    print(f"   HAM10000元数据记录: {len(ham_metadata)}")
                    print(f"   列名: {ham_metadata.columns.tolist()}")

                    # 处理HAM10000数据
                    for _, row in ham_metadata.iterrows():
                        image_id = row['image_id']
                        diagnosis = row['dx']  # HAM10000使用dx列
                        split = row['split'] if 'split' in row else 'train'

                        # HAM10000的标签映射
                        ham_label_mapping = {
                            'akiec': 0, 'bcc': 1, 'bkl': 2,
                            'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
                        }
                        numeric_label = ham_label_mapping.get(diagnosis.lower(), 5)

                        # 检查图像文件是否存在
                        image_path = os.path.join(ham_img_path, f"{image_id}.jpg")
                        if os.path.exists(image_path):
                            sample = {
                                'image_path': image_path,
                                'label': numeric_label,
                                'image_id': image_id
                            }

                            # 根据split分配
                            if split == 'train':
                                train_data.append(sample)
                            elif split == 'valid' or split == 'val':
                                val_data.append(sample)
                            elif split == 'test':
                                test_data.append(sample)

                    print(f"   HAM10000成功加载:")
                    print(f"      训练集: {len(train_data)} 样本")
                    print(f"      验证集: {len(val_data)} 样本")
                    print(f"      测试集: {len(test_data)} 样本")
                    ham_loaded = True
                    break

                except Exception as e:
                    print(f"   HAM10000加载失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        if not ham_loaded:
            print("   HAM10000数据集未找到")
            print(f"      尝试的路径:")
            for p in ham_paths_to_try:
                print(f"        - {p} (存在: {os.path.exists(p)})")
            for p in ham_metadata_paths_to_try:
                print(f"        - {p} (存在: {os.path.exists(p)})")

    # 检查是否加载成功
    if len(train_data) == 0 and len(val_data) == 0 and len(test_data) == 0:
        print("未找到有效数据")
        return None, None, None

    # 如果验证集为空，从训练集分割
    if len(val_data) == 0 and len(train_data) > 0:
        print("   验证集为空，从训练集分割15%作为验证集")
        np.random.shuffle(train_data)
        val_size = int(len(train_data) * 0.15)
        val_data = train_data[:val_size]
        train_data = train_data[val_size:]

    print(f"{dataset_name} 最终数据统计:")
    print(f"   训练集: {len(train_data)} 样本")
    print(f"   验证集: {len(val_data)} 样本")
    print(f"   测试集: {len(test_data)} 样本")

    return train_data, val_data, test_data

def main():
    """主函数"""
    config = AblationConfig()

    # 初始化日志文件
    with open(config.log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SwinTransformer消融实验 - 完整4步实验序列\n")
        f.write("=" * 80 + "\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结果目录: {config.results_dir}\n")
        f.write(f"模型目录: {config.models_dir}\n")
        f.write(f"实验数量: {len(config.experiments)}\n")
        f.write(f"训练轮数: {config.epochs}\n")
        f.write(f"批次大小: {config.batch_size}\n")
        f.write("=" * 80 + "\n\n")

    print(f"日志文件: {config.log_file}")

    # 数据集列表
    datasets = ['BCN20000', 'HAM10000']

    # 存储所有结果
    all_dataset_results = {}

    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"开始在 {dataset_name} 数据集上进行消融实验")
        print(f"{'='*80}")

        # 加载数据
        train_samples, val_samples, test_samples = load_dataset(dataset_name)
        if train_samples is None:
            print(f"{dataset_name} 数据加载失败，跳过")
            continue

        # 创建数据变换
        train_transform, val_transform = create_melanoma_focused_transforms()

        # 创建数据集
        train_dataset = MelanomaDataset(train_samples, train_transform, is_training=True)
        val_dataset = MelanomaDataset(val_samples, val_transform, is_training=False)

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers
        )

        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")

        # 加载之前完成的实验结果
        previous_results = load_previous_results(dataset_name)
        baseline_acc = previous_results[0]['best_accuracy']  # 获取真实baseline准确率

        print(f"\n加载之前完成的实验:")
        for prev_result in previous_results:
            print(f"   {prev_result['name']}: {prev_result['best_accuracy']:.2f}%")

        # 按优先级排序实验（从弱到强）
        sorted_experiments = sorted(
            config.experiments.items(),
            key=lambda x: x[1]['priority'],
            reverse=True
        )

        # 运行当前实验
        current_results = []

        for exp_name, exp_config in sorted_experiments:
            results = run_single_experiment(
                exp_name,
                exp_config,
                config,
                train_loader,
                val_loader,
                dataset_name,
                baseline_acc  # 传入真实baseline准确率
            )
            current_results.append(results)

            # 保存中间结果
            result_file = os.path.join(config.results_dir, f'{dataset_name}_{exp_name}_results.json')
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)

        # 合并之前和当前的结果
        all_results = previous_results + current_results

        # 保存该数据集的完整结果
        all_dataset_results[dataset_name] = all_results

        # 生成该数据集的对比报告（使用完整结果）
        generate_comparison_report(all_results, config, dataset_name)

    # 生成最终汇总报告
    generate_final_summary(all_dataset_results, config)

    print(f"\n所有实验完成!")
    print(f"结果保存在: {config.results_dir}")

def generate_comparison_report(results, config, dataset_name):
    """生成单个数据集的对比报告"""
    print(f"\n{'='*80}")
    print(f"{dataset_name} 消融实验对比报告")
    print(f"{'='*80}\n")

    # 创建DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('best_accuracy')

    # 打印表格
    print(df[['name', 'best_accuracy', 'melanoma_f1', 'improvement']].to_string(index=False))

    # 保存CSV
    csv_path = os.path.join(config.results_dir, f'{dataset_name}_ablation_comparison.csv')
    df.to_csv(csv_path, index=False)

    # 保存JSON
    json_path = os.path.join(config.results_dir, f'{dataset_name}_ablation_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 创新点贡献分析
    print(f"\n{'='*80}")
    print(f"{dataset_name} 创新点贡献分析")
    print(f"{'='*80}\n")

    baseline_acc = results[0]['best_accuracy']

    for i, result in enumerate(results[1:], 1):
        prev_acc = results[i-1]['best_accuracy']
        improvement = result['best_accuracy'] - prev_acc
        print(f"{result['name']}")
        print(f"  提升: +{improvement:.2f}% (vs 上一步)")
        print(f"  累计: +{result['improvement']:.2f}% (vs 基线)\n")

    print(f"{dataset_name} 总提升: +{results[-1]['improvement']:.2f}%")

def generate_final_summary(all_dataset_results, config):
    """生成最终汇总报告"""
    print(f"\n{'='*80}")
    print("最终汇总报告 - 双数据集消融实验")
    print(f"{'='*80}\n")

    summary_data = []

    for dataset_name, results in all_dataset_results.items():
        for result in results:
            summary_data.append({
                'Dataset': dataset_name,
                'Experiment': result['name'],
                'Accuracy': result['best_accuracy'],
                'F1_Macro': result['f1_macro'],
                'F1_Weighted': result['f1_weighted'],
                'MEL_F1': result['melanoma_f1'],
                'Improvement': result['improvement']
            })

    # 创建DataFrame
    df = pd.DataFrame(summary_data)

    # 打印完整表格
    print("所有实验结果:")
    print(df.to_string(index=False))

    # 保存CSV
    csv_path = os.path.join(config.results_dir, 'final_summary.csv')
    df.to_csv(csv_path, index=False)

    # 保存JSON
    json_path = os.path.join(config.results_dir, 'final_summary.json')
    with open(json_path, 'w') as f:
        json.dump(all_dataset_results, f, indent=2)

    # 统计分析
    print(f"\n{'='*80}")
    print("统计分析")
    print(f"{'='*80}\n")

    # 按实验分组统计
    exp_name_mapping = {
        'baseline_vit_ce': 'Baseline',
        'vit_focal': 'Exp1',
        'swin_focal': 'Exp2',
        'swin_dual_simple': 'Exp3',
        'swin_dual_attention': 'Exp4'
    }

    for _, exp_search in exp_name_mapping.items():
        exp_results = df[df['Experiment'].str.contains(exp_search, case=False)]
        if len(exp_results) > 0:
            avg_acc = exp_results['Accuracy'].mean()
            avg_mel_f1 = exp_results['MEL_F1'].mean()
            print(f"{exp_results.iloc[0]['Experiment']}:")
            print(f"  平均准确率: {avg_acc:.2f}%")
            print(f"  平均黑色素瘤F1: {avg_mel_f1:.4f}\n")

    # 最佳结果
    best_overall = df.loc[df['Accuracy'].idxmax()]
    best_mel = df.loc[df['MEL_F1'].idxmax()]

    print(f"最佳整体准确率:")
    print(f"   数据集: {best_overall['Dataset']}")
    print(f"   实验: {best_overall['Experiment']}")
    print(f"   准确率: {best_overall['Accuracy']:.2f}%\n")

    print(f"最佳黑色素瘤F1:")
    print(f"   数据集: {best_mel['Dataset']}")
    print(f"   实验: {best_mel['Experiment']}")
    print(f"   F1分数: {best_mel['MEL_F1']:.4f}\n")

    # 创新点平均贡献
    print(f"{'='*80}")
    print("创新点平均贡献（跨数据集）")
    print(f"{'='*80}\n")

    # 计算各个实验的平均提升
    baseline_improvement = df[df['Experiment'].str.contains('Baseline', case=False)]['Improvement'].mean()
    exp1_improvement = df[df['Experiment'].str.contains('Exp1', case=False)]['Improvement'].mean()
    exp2_improvement = df[df['Experiment'].str.contains('Exp2', case=False)]['Improvement'].mean()
    exp3_improvement = df[df['Experiment'].str.contains('Exp3', case=False)]['Improvement'].mean()
    exp4_improvement = df[df['Experiment'].str.contains('Exp4', case=False)]['Improvement'].mean()

    innovations = {
        'Focal Loss': exp1_improvement - baseline_improvement,
        'Swin架构': exp2_improvement - exp1_improvement,
        '双分支结构': exp3_improvement - exp2_improvement,
        '注意力融合': exp4_improvement - exp3_improvement
    }

    for innovation, contribution in innovations.items():
        if not np.isnan(contribution):
            print(f"{innovation}: +{contribution:.2f}%")

    print(f"\n最终汇总报告已保存:")
    print(f"   CSV: {csv_path}")
    print(f"   JSON: {json_path}")

if __name__ == '__main__':
    main()

