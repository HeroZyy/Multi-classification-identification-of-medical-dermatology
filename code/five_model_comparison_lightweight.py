"""
五组模型对比实验 - 轻量级简洁版（增强稳定性）
对BCN20000和HAM10000两个数据集分别进行训练和统计
包含：错误处理、日志保存、断点续训、进度监控
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
import timm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from datetime import datetime
import warnings
import logging
import json
import traceback
import time
warnings.filterwarnings('ignore')

# 导入数据处理模块
from melanoma_focused_improvement import (
    MelanomaDataset, create_melanoma_focused_transforms,
    FocalLoss, DISEASE_NAMES
)

def setup_logger(output_dir):
    """设置日志系统"""
    log_file = os.path.join(output_dir, 'training.log')

    # 创建logger
    logger = logging.getLogger('FiveModelComparison')
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class Config:
    """增强配置（含稳定性优化）"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 40
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4  # 与消融实验一致
        self.num_classes = 7
        self.num_workers = 8

        # 早停配置 - 已禁用，让所有模型跑满40轮
        self.use_early_stopping = False  # 🔴 禁用早停
        self.patience = 7
        self.min_delta = 0.001

        # Swin模型专用配置（与消融实验一致）
        self.use_cosine_scheduler = True  # Swin使用余弦退火
        self.use_gradient_clipping = True  # 使用梯度裁剪
        self.max_grad_norm = 1.0  # 梯度裁剪阈值

        # 五组模型配置 - 根据需求选择性训练
        # BCN20000: ViT_Base
        # HAM10000: DenseNet121, EfficientNet_B4, ViT_Base
        self.models = {
            'ViT_Base': 'vit_base_patch16_224',
            'EfficientNet_B4': 'efficientnet_b4',
            'ResNet50': 'resnet50',  
            'DenseNet121': 'densenet121',
            # 'Swin_Base': 'swin_base_patch4_window7_224'  
        }

        # 数据集-模型映射：指定每个数据集训练哪些模型
        self.dataset_models = {
            'BCN20000': ['ViT_Base'],  # BCN20000只训练ViT_Base
            'HAM10000': ['DenseNet121', 'EfficientNet_B4', 'ViT_Base']  # HAM10000训练这三个
        }

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f'results/five_model_comparison_{timestamp}'
        os.makedirs(self.output_dir, exist_ok=True)

        # 创建models子目录，为每个模型创建单独文件夹
        self.models_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        for model_name in self.models.keys():
            model_folder = os.path.join(self.models_dir, model_name)
            os.makedirs(model_folder, exist_ok=True)

        # 设置日志
        self.logger = setup_logger(self.output_dir)

        # 保存配置
        self.save_config()

    def save_config(self):
        """保存配置到文件"""
        config_dict = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'models': self.models,
            'device': str(self.device)
        }
        config_file = os.path.join(self.output_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

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
        # ============================================================
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def load_dataset(dataset_name):
    """加载数据集（与评估脚本保持一致）"""
    from sklearn.model_selection import train_test_split

    print(f"\n加载 {dataset_name} 数据集...")
    all_data = []

    if dataset_name == 'BCN20000':
        bcn_path = 'datasets/BCN20000'
        metadata = pd.read_csv(os.path.join(bcn_path, 'bcn20000.csv'))

        class_mapping = {
            'melanoma': 'MEL', 'nevus': 'NV', 'basal cell carcinoma': 'BCC',
            'seborrheic keratosis': 'BKL', 'actinic keratosis': 'AKIEC',
            'dermatofibroma': 'DF', 'vascular lesion': 'VASC'
        }
        label_mapping = {'AKIEC': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'VASC': 6}

        for _, row in metadata.iterrows():
            image_path = os.path.join(bcn_path, 'images', f"{row['isic_id']}.JPG")
            if os.path.exists(image_path):
                label = class_mapping.get(row['diagnosis'], 'NV')
                all_data.append({
                    'image_path': image_path,
                    'label': label_mapping.get(label, 5),
                    'image_id': row['isic_id']
                })

    elif dataset_name == 'HAM10000':
        ham_path = 'datasets/HAM10000_clean/ISIC2018'
        metadata = pd.read_csv('datasets/HAM10000_clean/ISIC2018_splits/HAM_clean.csv')

        label_mapping = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

        for _, row in metadata.iterrows():
            image_path = os.path.join(ham_path, f"{row['image_id']}.jpg")
            if os.path.exists(image_path):
                all_data.append({
                    'image_path': image_path,
                    'label': label_mapping.get(row['dx'].lower(), 5),
                    'image_id': row['image_id']
                })

    # 数据分割 (80% train, 10% val, 10% test) - 与评估脚本保持一致
    # 使用固定随机种子确保每次划分相同
    train_val, test_data = train_test_split(all_data, test_size=0.1, random_state=42)
    train_data, val_data = train_test_split(train_val, test_size=0.111, random_state=42)  # 0.111 * 0.9 ≈ 0.1

    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")
    print(f"⚠️  数据划分已更新为: 80% train / 10% val / 10% test (random_state=42)")
    return train_data, val_data, test_data

def train_model(model, train_loader, val_loader, config, model_name):
    """训练模型（含早停和错误处理）"""
    logger = config.logger

    try:
        model.to(config.device)

        # 提取基础模型名称（去掉数据集后缀）
        base_model_name = model_name.rsplit('_', 1)[0] if '_' in model_name else model_name

        # 根据模型类型配置优化器和调度器
        if 'Swin' in base_model_name:
            # Swin模型使用与消融实验一致的配置
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs
            )
            logger.info(f"🎯 {base_model_name} 使用Swin专用配置 (AdamW + CosineAnnealing + 梯度裁剪)")
        else:
            # 其他模型使用标准配置
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            scheduler = None
            logger.info(f"🎯 {base_model_name} 使用标准配置 (AdamW)")

        criterion = FocalLoss(alpha=0.25, gamma=2.0)

        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()

        logger.info(f"开始训练 {model_name}")

        for epoch in range(config.epochs):
            epoch_start = time.time()

            # 训练
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            try:
                for images, labels, _ in tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{config.epochs}'):
                    images, labels = images.to(config.device), labels.to(config.device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    # Swin模型使用梯度裁剪（与消融实验一致）
                    if 'Swin' in base_model_name and config.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)

                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
            except Exception as e:
                logger.error(f"训练循环错误: {str(e)}")
                logger.error(traceback.format_exc())
                raise

            # 验证
            model.eval()
            val_preds, val_labels = [], []

            try:
                with torch.no_grad():
                    for images, labels, _ in val_loader:
                        images = images.to(config.device)
                        outputs = model(images)
                        _, predicted = outputs.max(1)
                        val_preds.extend(predicted.cpu().numpy())
                        val_labels.extend(labels.numpy())
            except Exception as e:
                logger.error(f"验证循环错误: {str(e)}")
                logger.error(traceback.format_exc())
                raise

            val_acc = accuracy_score(val_labels, val_preds) * 100
            train_acc = 100. * train_correct / train_total
            epoch_time = time.time() - epoch_start

            # 更新学习率调度器（Swin模型）
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                log_msg = f'Epoch {epoch+1}/{config.epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {current_lr:.2e}, Time: {epoch_time:.1f}s'
            else:
                log_msg = f'Epoch {epoch+1}/{config.epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Time: {epoch_time:.1f}s'

            logger.info(log_msg)
            print(log_msg)

            # 保存最佳模型到对应的模型文件夹
            if val_acc > best_val_acc + config.min_delta:
                best_val_acc = val_acc
                patience_counter = 0

                # 提取模型名称（去掉数据集后缀）
                base_model_name = model_name.rsplit('_', 1)[0]  # 例如: ViT_Base_BCN20000 -> ViT_Base
                model_folder = os.path.join(config.models_dir, base_model_name)
                checkpoint_path = os.path.join(model_folder, f'{model_name}_best.pth')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'model_name': model_name,
                    'base_model_name': base_model_name,
                }, checkpoint_path)
                logger.info(f'保存最佳模型: {best_val_acc:.2f}% -> {checkpoint_path}')
            else:
                patience_counter += 1
                logger.info(f'无提升 ({patience_counter}/{config.patience})')

            # 早停机制 - 已禁用
            if config.use_early_stopping and patience_counter >= config.patience:
                logger.info(f'早停触发 at epoch {epoch+1}')
                break

        total_time = time.time() - start_time
        logger.info(f'{model_name} 训练完成! 最佳准确率: {best_val_acc:.2f}%, 总时间: {total_time/60:.1f}分钟')

        return model, best_val_acc

    except Exception as e:
        logger.error(f"训练 {model_name} 时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        # 保存错误状态
        error_file = os.path.join(config.output_dir, f'{model_name}_error.txt')
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"错误时间: {datetime.now()}\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"堆栈跟踪:\n{traceback.format_exc()}")
        raise

def evaluate_model(model, test_loader, config):
    """评估模型"""
    model.eval()
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(config.device)
            outputs = model(images)
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

def train_on_dataset(dataset_name, config):
    """在指定数据集上训练所有模型（含错误恢复）"""
    logger = config.logger

    logger.info(f"\n{'='*60}")
    logger.info(f"开始在 {dataset_name} 上训练五组模型")
    logger.info(f"{'='*60}")

    try:
        # 加载数据
        train_data, val_data, test_data = load_dataset(dataset_name)

        # 创建数据加载器
        train_transform, val_transform = create_melanoma_focused_transforms()
        train_dataset = MelanomaDataset(train_data, train_transform, is_training=True)
        val_dataset = MelanomaDataset(val_data, val_transform, is_training=False)
        test_dataset = MelanomaDataset(test_data, val_transform, is_training=False)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                 num_workers=config.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                               num_workers=config.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, pin_memory=True)

        # 训练指定的模型（根据数据集）
        results = {}
        progress_file = os.path.join(config.output_dir, f'{dataset_name}_progress.json')

        # 获取当前数据集需要训练的模型列表
        models_to_train = config.dataset_models.get(dataset_name, list(config.models.keys()))
        logger.info(f"📋 {dataset_name} 将训练以下模型: {models_to_train}")

        # 过滤出需要训练的模型
        filtered_models = {k: v for k, v in config.models.items() if k in models_to_train}

        for idx, (model_name, backbone_name) in enumerate(filtered_models.items(), 1):
            logger.info(f"\n[{idx}/{len(filtered_models)}] 训练 {model_name}...")

            try:
                # 创建模型
                model = SimpleModel(backbone_name, config.num_classes)

                # 训练
                model, best_val_acc = train_model(model, train_loader, val_loader, config,
                                                 f'{model_name}_{dataset_name}')

                # 加载最佳权重（从模型文件夹中）
                model_folder = os.path.join(config.models_dir, model_name)
                checkpoint_path = os.path.join(model_folder, f'{model_name}_{dataset_name}_best.pth')
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])

                # 评估
                result = evaluate_model(model, test_loader, config)
                result['best_val_acc'] = best_val_acc
                results[model_name] = result

                logger.info(f"✅ {model_name} 完成: ACC={result['accuracy']:.2f}%, MEL F1={result['melanoma_f1']:.3f}")

                # 保存进度
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

                # 清理显存
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"训练 {model_name} 失败: {str(e)}")
                logger.error(traceback.format_exc())
                results[model_name] = {'error': str(e), 'accuracy': 0.0}
                # 继续训练下一个模型
                continue

        return results

    except Exception as e:
        logger.error(f"数据集 {dataset_name} 训练失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def save_results(all_results, config):
    """保存结果（增强版 - 分数据集统计）"""
    logger = config.logger

    # 1. 保存完整结果表格
    data = []
    for dataset, results in all_results.items():
        for model, metrics in results.items():
            if 'error' in metrics:
                data.append({
                    'Dataset': dataset,
                    'Model': model,
                    'Accuracy': 'ERROR',
                    'Macro_F1': 'ERROR',
                    'Weighted_F1': 'ERROR',
                    'Melanoma_F1': 'ERROR',
                    'Error': metrics['error']
                })
            else:
                data.append({
                    'Dataset': dataset,
                    'Model': model,
                    'Accuracy': f"{metrics['accuracy']:.2f}%",
                    'Macro_F1': f"{metrics['macro_f1']:.3f}",
                    'Weighted_F1': f"{metrics['weighted_f1']:.3f}",
                    'Melanoma_F1': f"{metrics['melanoma_f1']:.3f}",
                    'Best_Val_Acc': f"{metrics.get('best_val_acc', 0):.2f}%"
                })

    df = pd.DataFrame(data)

    # 保存完整结果CSV
    csv_path = os.path.join(config.output_dir, 'results_complete.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"完整结果已保存到: {csv_path}")

    # 2. 分数据集保存结果
    for dataset in all_results.keys():
        dataset_df = df[df['Dataset'] == dataset].copy()
        dataset_csv = os.path.join(config.output_dir, f'results_{dataset}.csv')
        dataset_df.to_csv(dataset_csv, index=False, encoding='utf-8-sig')
        logger.info(f"{dataset} 结果已保存到: {dataset_csv}")

    # 3. 生成统计摘要
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("五组模型对比实验 - 结果统计摘要")
    summary_lines.append("="*80)
    summary_lines.append("")

    for dataset, results in all_results.items():
        summary_lines.append(f"\n{'#'*80}")
        summary_lines.append(f"# 数据集: {dataset}")
        summary_lines.append(f"{'#'*80}")
        summary_lines.append("")

        # 按准确率排序
        sorted_results = sorted(
            [(model, metrics) for model, metrics in results.items() if 'error' not in metrics],
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        summary_lines.append(f"{'模型':<20} {'准确率':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Melanoma F1':<12}")
        summary_lines.append("-"*80)

        for rank, (model, metrics) in enumerate(sorted_results, 1):
            summary_lines.append(
                f"{rank}. {model:<17} "
                f"{metrics['accuracy']:>6.2f}%     "
                f"{metrics['macro_f1']:>6.3f}       "
                f"{metrics['weighted_f1']:>6.3f}         "
                f"{metrics['melanoma_f1']:>6.3f}"
            )

        # 统计信息
        if sorted_results:
            accuracies = [m['accuracy'] for _, m in sorted_results]
            mel_f1s = [m['melanoma_f1'] for _, m in sorted_results]

            summary_lines.append("")
            summary_lines.append(f"统计信息:")
            summary_lines.append(f"  - 最佳准确率: {max(accuracies):.2f}% ({sorted_results[0][0]})")
            summary_lines.append(f"  - 最差准确率: {min(accuracies):.2f}% ({sorted_results[-1][0]})")
            summary_lines.append(f"  - 平均准确率: {np.mean(accuracies):.2f}%")
            summary_lines.append(f"  - 最佳Melanoma F1: {max(mel_f1s):.3f}")
            summary_lines.append(f"  - 平均Melanoma F1: {np.mean(mel_f1s):.3f}")

    # 4. 跨数据集对比
    summary_lines.append(f"\n{'#'*80}")
    summary_lines.append(f"# 跨数据集对比")
    summary_lines.append(f"{'#'*80}")
    summary_lines.append("")

    for model_name in config.models.keys():
        summary_lines.append(f"\n{model_name}:")
        for dataset in all_results.keys():
            if model_name in all_results[dataset] and 'error' not in all_results[dataset][model_name]:
                metrics = all_results[dataset][model_name]
                summary_lines.append(
                    f"  {dataset:<12}: ACC={metrics['accuracy']:>6.2f}%, "
                    f"MEL F1={metrics['melanoma_f1']:.3f}"
                )

    summary_lines.append("")
    summary_lines.append("="*80)

    # 保存摘要
    summary_text = '\n'.join(summary_lines)
    summary_path = os.path.join(config.output_dir, 'RESULTS_SUMMARY.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    logger.info(f"统计摘要已保存到: {summary_path}")

    # 打印摘要
    print("\n" + summary_text)

    # 5. 保存模型文件夹信息
    models_info = []
    models_info.append("="*80)
    models_info.append("模型文件保存位置")
    models_info.append("="*80)
    models_info.append("")

    for model_name in config.models.keys():
        model_folder = os.path.join(config.models_dir, model_name)
        models_info.append(f"\n{model_name}:")
        models_info.append(f"  文件夹: {model_folder}")

        # 列出该文件夹中的文件
        if os.path.exists(model_folder):
            files = os.listdir(model_folder)
            if files:
                models_info.append(f"  包含文件:")
                for f in sorted(files):
                    file_path = os.path.join(model_folder, f)
                    file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                    models_info.append(f"    - {f} ({file_size:.1f} MB)")
            else:
                models_info.append(f"  (空文件夹)")

    models_info.append("")
    models_info.append("="*80)

    models_info_text = '\n'.join(models_info)
    models_info_path = os.path.join(config.output_dir, 'MODELS_INFO.txt')
    with open(models_info_path, 'w', encoding='utf-8') as f:
        f.write(models_info_text)

    logger.info(f"模型信息已保存到: {models_info_path}")
    print("\n" + models_info_text)

def main():
    """主函数（含完整错误处理）"""
    start_time = time.time()

    try:
        print("="*80)
        print("五组模型对比实验 - 增强稳定版")
        print("包含：错误处理、日志保存、断点续训、进度监控")
        print("="*80)

        config = Config()
        logger = config.logger

        logger.info(f"设备: {config.device}")
        logger.info(f"输出目录: {config.output_dir}")
        early_stop_status = "禁用 (跑满40轮)" if not config.use_early_stopping else f"启用 (Patience={config.patience})"
        logger.info(f"训练配置: Epochs={config.epochs}, Batch={config.batch_size}, 早停={early_stop_status}")
        logger.info(f"BCN20000训练模型: {config.dataset_models['BCN20000']}")
        logger.info(f"HAM10000训练模型: {config.dataset_models['HAM10000']}")

        # 训练两个数据集
        all_results = {}
        dataset_times = {}

        for dataset in ['BCN20000', 'HAM10000']:
            dataset_start = time.time()
            logger.info(f"\n{'#'*80}")
            logger.info(f"# 数据集: {dataset}")
            logger.info(f"{'#'*80}")

            try:
                results = train_on_dataset(dataset, config)
                all_results[dataset] = results
                dataset_time = time.time() - dataset_start
                dataset_times[dataset] = dataset_time
                logger.info(f"{dataset} 完成! 用时: {dataset_time/60:.1f}分钟")
            except Exception as e:
                logger.error(f"{dataset} 训练失败: {str(e)}")
                logger.error(traceback.format_exc())
                all_results[dataset] = {'error': str(e)}
                # 继续下一个数据集
                continue

        # 保存结果
        try:
            save_results(all_results, config)
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
            logger.error(traceback.format_exc())

        # 总结
        total_time = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info("✅ 所有实验完成!")
        logger.info(f"总用时: {total_time/3600:.2f}小时 ({total_time/60:.1f}分钟)")
        for dataset, dt in dataset_times.items():
            logger.info(f"  - {dataset}: {dt/60:.1f}分钟")
        logger.info(f"结果保存在: {config.output_dir}")
        logger.info("="*80)

        return 0

    except Exception as e:
        print(f"\n❌ 程序发生严重错误: {str(e)}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

