"""

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import timm
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 疾病映射
DISEASE_MAPPING = {
    'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
}

DISEASE_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

class MelanomaFocusedConfig:
    """黑色素瘤专项优化配置"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 基础配置
        self.image_size = 224
        self.batch_size = 16
        self.epochs = 50
        self.num_classes = 7
        
        # 黑色素瘤专项配置
        self.melanoma_class_idx = 4  # MEL在映射中的索引
        self.melanoma_weight_multiplier = 3.0  # 黑色素瘤权重倍数
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        
        # 学习率策略
        self.base_lr = 1e-4
        self.melanoma_lr_multiplier = 2.0  # 黑色素瘤相关层使用更高学习率
        
        # 数据增强
        self.use_advanced_augmentation = True
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0

class MelanomaDataset(Dataset):
    """黑色素瘤专项数据集"""
    def __init__(self, samples, transform=None, is_training=False):
        self.samples = samples
        self.transform = transform
        self.is_training = is_training
        
        # 统计各类别数量
        self.class_counts = {}
        for sample in samples:
            label = sample['label']
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
        
        print(f"数据集类别分布: {self.class_counts}")
        if 4 in self.class_counts:
            print(f"黑色素瘤样本数量: {self.class_counts[4]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label'], sample['image_id']

def create_melanoma_focused_transforms():
    """创建黑色素瘤专项数据增强"""
    
    # 训练时的强化增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # 增加垂直翻转
        transforms.RandomRotation(20, interpolation=transforms.InterpolationMode.BILINEAR),
        
        # 颜色增强 - 对黑色素瘤特别重要
        transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1
        ),
        
        # 高斯模糊
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.2),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # 随机擦除
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class FocalLoss(nn.Module):
    """Focal Loss - 解决类别不平衡问题"""
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
        # 为黑色素瘤设置更高的alpha值
        if isinstance(alpha, (float, int)):
            self.alpha = torch.ones(num_classes) * alpha
            self.alpha[4] = alpha * 2.0  # 黑色素瘤使用2倍alpha
        else:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # 获取alpha值
        alpha_t = self.alpha.to(inputs.device)[targets]

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class MelanomaFocusedModel(nn.Module):
    """黑色素瘤专项优化模型"""
    
    def __init__(self, config, model_name='efficientnet_b4'):
        super().__init__()
        self.config = config
        self.model_name = model_name
        
        # 创建骨干网络
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        
        # 黑色素瘤专项特征提取器
        self.melanoma_feature_extractor = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        
        # 通用分类器
        self.general_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, config.num_classes)
        )
        
        # 黑色素瘤专项分类器
        self.melanoma_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 二分类：是否为黑色素瘤
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.num_classes + 2, config.num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x, return_features=False):
        # 特征提取
        features = self.backbone(x)
        
        # 通用分类
        general_logits = self.general_classifier(features)
        
        # 黑色素瘤专项处理
        melanoma_features = self.melanoma_feature_extractor(features)
        melanoma_logits = self.melanoma_classifier(melanoma_features)
        
        # 特征融合
        combined_features = torch.cat([general_logits, melanoma_logits], dim=1)
        final_logits = self.fusion_layer(combined_features)
        
        if return_features:
            return final_logits, features, melanoma_features
        else:
            return final_logits

def create_weighted_sampler(samples):
    """创建加权采样器，重点关注黑色素瘤"""
    labels = [sample['label'] for sample in samples]
    class_counts = np.bincount(labels, minlength=7)
    
    # 计算权重，黑色素瘤给予额外权重
    weights = 1.0 / class_counts
    weights[4] *= 2.0  # 黑色素瘤额外2倍权重
    
    sample_weights = [weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def load_and_prepare_data():
    """加载和准备数据"""
    csv_file = r"HAM\HAM10000_clean\HAM10000_clean\ISIC2018_splits\HAM_clean.csv"
    image_dir = r"HAM\HAM10000_clean\HAM10000_clean\ISIC2018"
    
    if not os.path.exists(csv_file) or not os.path.exists(image_dir):
        print("❌ 数据文件不存在")
        return None, None, None
    
    df = pd.read_csv(csv_file)
    
    # 准备样本
    samples = []
    melanoma_count = 0
    
    for idx, row in df.iterrows():
        image_path = os.path.join(image_dir, row['image'])
        if os.path.exists(image_path):
            label = DISEASE_MAPPING[row['dx']]
            samples.append({
                'image_path': image_path,
                'image_id': row['image_id'],
                'label': label,
                'disease': row['dx']
            })
            
            if label == 4:  # 黑色素瘤
                melanoma_count += 1
    
    print(f"📊 总样本数: {len(samples)}")
    print(f"🔍 黑色素瘤样本数: {melanoma_count} ({melanoma_count/len(samples)*100:.1f}%)")
    
    # 分层分割，确保黑色素瘤在各集合中都有足够样本
    labels = [sample['label'] for sample in samples]
    train_samples, temp_samples = train_test_split(
        samples, test_size=0.3, stratify=labels, random_state=42
    )
    
    temp_labels = [sample['label'] for sample in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    return train_samples, val_samples, test_samples

def train_melanoma_focused_model(model, train_loader, val_loader, config):
    """训练黑色素瘤专项模型"""
    model.to(config.device)
    
    # 优化器 - 对黑色素瘤相关层使用更高学习率
    melanoma_params = list(model.melanoma_feature_extractor.parameters()) + \
                     list(model.melanoma_classifier.parameters())
    general_params = list(model.backbone.parameters()) + \
                    list(model.general_classifier.parameters()) + \
                    list(model.fusion_layer.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': general_params, 'lr': config.base_lr},
        {'params': melanoma_params, 'lr': config.base_lr * config.melanoma_lr_multiplier}
    ], weight_decay=0.01)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # 损失函数
    criterion = FocalLoss(alpha=config.focal_loss_alpha, gamma=config.focal_loss_gamma)
    
    print(f"🚀 开始黑色素瘤专项训练")
    print(f"📊 配置: Epochs={config.epochs}, Base LR={config.base_lr}")
    
    best_val_acc = 0.0
    best_melanoma_f1 = 0.0
    
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        for images, labels, _ in pbar:
            images, labels = images.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # 验证阶段
        model.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(config.device), labels.to(config.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        val_acc = accuracy_score(val_labels, val_predictions) * 100
        
        # 计算黑色素瘤F1分数
        from sklearn.metrics import f1_score
        melanoma_f1 = f1_score(
            [1 if l == 4 else 0 for l in val_labels],
            [1 if p == 4 else 0 for p in val_predictions],
            average='binary'
        )
        
        scheduler.step()
        
        train_acc = 100. * train_correct / train_total
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, MEL F1: {melanoma_f1:.3f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc or melanoma_f1 > best_melanoma_f1:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if melanoma_f1 > best_melanoma_f1:
                best_melanoma_f1 = melanoma_f1
            
            torch.save(model.state_dict(), f'melanoma_focused_{model.model_name}.pth')
            print(f"🎯 保存最佳模型: Val Acc={best_val_acc:.2f}%, MEL F1={best_melanoma_f1:.3f}")
    
    return model, best_val_acc, best_melanoma_f1

def main():
    """主函数"""
    print("🎯 黑色素瘤专项改进训练")
    print("=" * 50)
    
    # 配置
    config = MelanomaFocusedConfig()
    print(f"📱 设备: {config.device}")
    
    # 加载数据
    train_samples, val_samples, test_samples = load_and_prepare_data()
    if train_samples is None:
        return
    
    # 创建数据变换
    train_transform, val_transform = create_melanoma_focused_transforms()
    
    # 创建数据集
    train_dataset = MelanomaDataset(train_samples, train_transform, is_training=True)
    val_dataset = MelanomaDataset(val_samples, val_transform, is_training=False)
    
    # 创建加权采样器
    weighted_sampler = create_weighted_sampler(train_samples)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=weighted_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = MelanomaFocusedModel(config, model_name='efficientnet_b4')
    
    # 训练模型
    trained_model, best_acc, best_mel_f1 = train_melanoma_focused_model(
        model, train_loader, val_loader, config
    )
    
    print(f"\n🎯 训练完成!")
    print(f"   - 最佳验证准确率: {best_acc:.2f}%")
    print(f"   - 最佳黑色素瘤F1: {best_mel_f1:.3f}")

if __name__ == "__main__":
    main()
