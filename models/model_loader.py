"""
SWIN消融实验模型加载工具
提供便捷的模型加载和推理接口
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
import timm
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
import json
from typing import Dict, Tuple, Optional, Union

class FocalLoss(nn.Module):
    """Focal Loss实现"""
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class SwinDualBranchSimpleModel(nn.Module):
    """Swin双分支简化模型"""
    def __init__(self, num_classes=7, mel_weight=0.3):
        super(SwinDualBranchSimpleModel, self).__init__()

        # ============================================================
        # 在线模式（需要网络连接，从 Hugging Face Hub 下载预训练权重）
        # ============================================================
        # self.backbone = timm.create_model('swin_base_patch4_window7_224',
        #                                 pretrained=True,
        #                                 num_classes=0)

        # ============================================================
        # 离线模式（不需要网络连接，只创建模型结构）
        # 用于加载已训练模型时，不需要下载 ImageNet 预训练权重
        # ============================================================
        self.backbone = timm.create_model('swin_base_patch4_window7_224',
                                        pretrained=False,
                                        num_classes=0)
        
        feature_dim = self.backbone.num_features
        self.main_classifier = nn.Linear(feature_dim, num_classes)
        self.mel_classifier = nn.Linear(feature_dim, 2)
        self.mel_weight = mel_weight
        
    def forward(self, x):
        features = self.backbone(x)
        main_logits = self.main_classifier(features)
        mel_logits = self.mel_classifier(features)
        return main_logits, mel_logits

class ModelLoader:
    """模型加载器"""
    
    def __init__(self, models_dir: str = "SWIN/models"):
        self.models_dir = Path(models_dir)
        self.class_names = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_vit_focal_model(self, dataset: str = "BCN20000", 
                           best: bool = True) -> torch.nn.Module:
        """加载ViT + Focal Loss模型"""
        model_type = "best" if best else "latest"
        model_path = self.models_dir / "vit_focal" / f"{dataset}_vit_focal_{model_type}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = timm.create_model('vit_base_patch16_224', 
                                pretrained=False, 
                                num_classes=7)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def load_swin_focal_model(self, dataset: str = "HAM10000", 
                            best: bool = True) -> torch.nn.Module:
        """加载Swin + Focal Loss模型"""
        model_type = "best" if best else "latest"
        model_path = self.models_dir / "swin_focal" / f"{dataset}_swin_focal_{model_type}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = timm.create_model('swin_base_patch4_window7_224', 
                                pretrained=False, 
                                num_classes=7)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def load_swin_dual_model(self, dataset: str = "HAM10000", 
                           best: bool = True) -> torch.nn.Module:
        """加载Swin双分支模型"""
        model_type = "best" if best else "latest"
        model_path = self.models_dir / "swin_dual_branch" / f"{dataset}_swin_dual_simple_{model_type}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = SwinDualBranchSimpleModel(num_classes=7)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def get_model_info(self, model_path: str) -> Dict:
        """获取模型信息"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'best_accuracy': checkpoint.get('best_accuracy', 'Unknown'),
            'optimizer_state': 'optimizer_state_dict' in checkpoint,
            'scheduler_state': 'scheduler_state_dict' in checkpoint,
        }
        
        return info
    
    def predict(self, model: torch.nn.Module, image_tensor: torch.Tensor, 
                device: str = 'cpu') -> Tuple[str, float, Dict]:
        """模型推理"""
        model = model.to(device)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            if isinstance(model, SwinDualBranchSimpleModel):
                main_logits, mel_logits = model(image_tensor.unsqueeze(0))
                probabilities = torch.softmax(main_logits, dim=1)
                mel_prob = torch.softmax(mel_logits, dim=1)
                
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
                mel_confidence = mel_prob[0][1].item()  # 黑色素瘤概率
                
                result = {
                    'all_probabilities': probabilities[0].cpu().numpy(),
                    'melanoma_probability': mel_confidence,
                    'dual_branch': True
                }
            else:
                logits = model(image_tensor.unsqueeze(0))
                probabilities = torch.softmax(logits, dim=1)
                
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
                
                result = {
                    'all_probabilities': probabilities[0].cpu().numpy(),
                    'dual_branch': False
                }
        
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, confidence, result
    
    def list_available_models(self) -> Dict:
        """列出所有可用模型"""
        models = {}
        
        for model_type in ['vit_focal', 'swin_focal', 'swin_dual_branch']:
            model_dir = self.models_dir / model_type
            if model_dir.exists():
                models[model_type] = []
                for model_file in model_dir.glob("*.pth"):
                    models[model_type].append(model_file.name)
        
        return models

# 使用示例
if __name__ == "__main__":
    # 初始化模型加载器
    loader = ModelLoader()
    
    # 列出可用模型
    print("可用模型:")
    available_models = loader.list_available_models()
    for model_type, files in available_models.items():
        print(f"  {model_type}: {files}")
    
    # 加载最佳Swin模型
    try:
        model = loader.load_swin_focal_model("HAM10000", best=True)
        print(f"\n成功加载模型: Swin + Focal Loss (HAM10000)")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建示例输入
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"输出形状: {output.shape}")
            
    except FileNotFoundError as e:
        print(f"错误: {e}")
    
    print("\n模型加载器初始化完成！")
