#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vision Transformer (ViT)系列模型定义

包含:
1. ViT_Base - Vision Transformer基线模型
2. ViT_Baseline - ViT基线模型（另一种配置）
3. ViT_Focal - ViT + Focal Loss模型

参考文献:
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020).
An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR.
"""

import torch
import torch.nn as nn
import timm


class ViTBaseModel(nn.Module):
    """
    Vision Transformer基线模型
    
    架构:
        - Backbone: ViT-Base (vit_base_patch16_224)
        - Classifier: Dropout + Linear + Dropout
    
    参数:
        num_classes (int): 分类类别数，默认7
        pretrained (bool): 是否使用预训练权重，默认True
        dropout (float): Dropout比率，默认0.6
    """
    
    def __init__(self, num_classes=7, pretrained=True, dropout=0.6):
        super().__init__()
        
        # 加载ViT backbone
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.backbone.head = nn.Identity()  # 移除原始分类头
        
        # 获取特征维度
        feature_dim = 768  # ViT-Base的特征维度
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'ViT_Base',
            'backbone': 'vit_base_patch16_224',
            'feature_dim': 768,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class ViTBaselineModel(nn.Module):
    """
    ViT基线模型（使用num_classes=0配置）
    
    架构:
        - Backbone: ViT-Base (vit_base_patch16_224, num_classes=0)
        - Classifier: Dropout + Linear + Dropout
    
    参数:
        num_classes (int): 分类类别数，默认7
        pretrained (bool): 是否使用预训练权重，默认True
        dropout (float): Dropout比率，默认0.6
    """
    
    def __init__(self, num_classes=7, pretrained=True, dropout=0.6):
        super().__init__()
        
        # 加载ViT backbone（使用num_classes=0）
        self.backbone = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=pretrained,
            num_classes=0  # 这会使用fc_norm而不是norm
        )
        
        # 获取特征维度
        feature_dim = self.backbone.num_features
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'ViT_Baseline',
            'backbone': 'vit_base_patch16_224 (num_classes=0)',
            'feature_dim': self.backbone.num_features,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class ViTFocalModel(nn.Module):
    """
    ViT + Focal Loss模型
    
    架构与ViTBaselineModel相同，但训练时使用Focal Loss
    
    参数:
        num_classes (int): 分类类别数，默认7
        pretrained (bool): 是否使用预训练权重，默认True
        dropout (float): Dropout比率，默认0.6
    """
    
    def __init__(self, num_classes=7, pretrained=True, dropout=0.6):
        super().__init__()
        
        # 加载ViT backbone
        self.backbone = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=pretrained,
            num_classes=0
        )
        
        # 获取特征维度
        feature_dim = self.backbone.num_features
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'ViT_Focal',
            'backbone': 'vit_base_patch16_224',
            'loss': 'Focal Loss',
            'feature_dim': self.backbone.num_features,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class DeepViTModel(nn.Module):
    """
    深度ViT模型（用于五模型对比实验）
    
    架构:
        - Backbone: ViT-Base
        - Classifier: Dropout + Linear + Dropout
    
    参数:
        num_classes (int): 分类类别数，默认7
        pretrained (bool): 是否使用预训练权重，默认True
        dropout (float): Dropout比率，默认0.1
    """
    
    def __init__(self, num_classes=7, pretrained=True, dropout=0.1):
        super().__init__()
        
        # 加载ViT backbone
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.backbone.head = nn.Identity()
        
        # 特征维度
        feature_dim = 768
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'DeepViT',
            'backbone': 'vit_base_patch16_224',
            'feature_dim': 768,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# 便捷函数
def get_vit_model(model_type='base', **kwargs):
    """
    获取ViT模型实例
    
    参数:
        model_type (str): 模型类型，可选'base', 'baseline', 'focal', 'deep'
        **kwargs: 传递给模型的参数
    
    返回:
        model: ViT模型实例
    """
    if model_type == 'base':
        return ViTBaseModel(**kwargs)
    elif model_type == 'baseline':
        return ViTBaselineModel(**kwargs)
    elif model_type == 'focal':
        return ViTFocalModel(**kwargs)
    elif model_type == 'deep':
        return DeepViTModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 示例用法
if __name__ == "__main__":
    # 测试ViT_Base
    print("=" * 50)
    print("ViT_Base模型")
    print("=" * 50)
    model = ViTBaseModel(num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试ViT_Baseline
    print("\n" + "=" * 50)
    print("ViT_Baseline模型")
    print("=" * 50)
    model = ViTBaselineModel(num_classes=7, pretrained=False)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试ViT_Focal
    print("\n" + "=" * 50)
    print("ViT_Focal模型")
    print("=" * 50)
    model = ViTFocalModel(num_classes=7, pretrained=False)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试DeepViT
    print("\n" + "=" * 50)
    print("DeepViT模型")
    print("=" * 50)
    model = DeepViTModel(num_classes=7, pretrained=False)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型信息: {model.get_model_info()}")

