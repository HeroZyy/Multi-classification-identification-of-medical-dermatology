#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型定义模块

提供所有模型的统一接口
"""

from .swin_models import SwinBaseModel, SwinFocalModel, SwinDualBranchModel, get_swin_model
from .vit_models import ViTBaseModel, ViTBaselineModel, ViTFocalModel, DeepViTModel, get_vit_model
from .cnn_models import EfficientNetB4Model, ResNet50Model, DenseNet121Model, ResNet152Model, get_cnn_model
from .focal_loss import FocalLoss, AdaptiveFocalLoss, MelanomaFocalLoss, get_focal_loss

__all__ = [
    # Swin模型
    'SwinBaseModel',
    'SwinFocalModel',
    'SwinDualBranchModel',
    'get_swin_model',
    
    # ViT模型
    'ViTBaseModel',
    'ViTBaselineModel',
    'ViTFocalModel',
    'DeepViTModel',
    'get_vit_model',
    
    # CNN模型
    'EfficientNetB4Model',
    'ResNet50Model',
    'DenseNet121Model',
    'ResNet152Model',
    'get_cnn_model',
    
    # 损失函数
    'FocalLoss',
    'AdaptiveFocalLoss',
    'MelanomaFocalLoss',
    'get_focal_loss',
    
    # 模型工厂
    'create_model',
    'get_model_list',
]


def create_model(model_name, num_classes=7, pretrained=True, **kwargs):
    """
    统一的模型创建接口
    
    参数:
        model_name (str): 模型名称
        num_classes (int): 分类类别数
        pretrained (bool): 是否使用预训练权重
        **kwargs: 其他模型参数
    
    返回:
        model: 模型实例
    
    支持的模型:
        - Swin系列: 'Swin_Base', 'Swin_Focal', 'Swin_Dual_Branch'
        - ViT系列: 'ViT_Base', 'ViT_Baseline', 'ViT_Focal', 'DeepViT'
        - CNN系列: 'EfficientNet_B4', 'ResNet50', 'DenseNet121', 'ResNet152'
    """
    model_name = model_name.lower()
    
    # Swin系列
    if model_name in ['swin_base', 'swinbase']:
        return SwinBaseModel(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name in ['swin_focal', 'swinfocal']:
        return SwinFocalModel(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name in ['swin_dual_branch', 'swindualbranch', 'swin_dual']:
        return SwinDualBranchModel(num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    # ViT系列
    elif model_name in ['vit_base', 'vitbase']:
        return ViTBaseModel(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name in ['vit_baseline', 'vitbaseline']:
        return ViTBaselineModel(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name in ['vit_focal', 'vitfocal']:
        return ViTFocalModel(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name in ['deepvit', 'deep_vit']:
        return DeepViTModel(num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    # CNN系列
    elif model_name in ['efficientnet_b4', 'efficientnetb4']:
        return EfficientNetB4Model(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name in ['resnet50']:
        return ResNet50Model(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name in ['densenet121']:
        return DenseNet121Model(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name in ['resnet152']:
        return ResNet152Model(num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_model_list():
    """
    获取所有支持的模型列表
    
    返回:
        dict: 模型分类字典
    """
    return {
        'Swin系列': [
            'Swin_Base',
            'Swin_Focal',
            'Swin_Dual_Branch'
        ],
        'ViT系列': [
            'ViT_Base',
            'ViT_Baseline',
            'ViT_Focal',
            'DeepViT'
        ],
        'CNN系列': [
            'EfficientNet_B4',
            'ResNet50',
            'DenseNet121',
            'ResNet152'
        ]
    }


# 示例用法
if __name__ == "__main__":
    import torch
    
    print("=" * 70)
    print("模型工厂测试")
    print("=" * 70)
    
    # 显示所有支持的模型
    print("\n支持的模型:")
    model_list = get_model_list()
    for category, models in model_list.items():
        print(f"\n{category}:")
        for model_name in models:
            print(f"  - {model_name}")
    
    # 测试创建几个模型
    print("\n" + "=" * 70)
    print("测试创建模型")
    print("=" * 70)
    
    test_models = ['Swin_Base', 'ViT_Base', 'EfficientNet_B4']
    x = torch.randn(2, 3, 224, 224)
    
    for model_name in test_models:
        print(f"\n创建模型: {model_name}")
        model = create_model(model_name, num_classes=7, pretrained=False)
        output = model(x)
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            print(f"  模型信息: {info}")

