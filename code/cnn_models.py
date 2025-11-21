#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN系列模型定义

包含:
1. EfficientNet_B4 - EfficientNet-B4模型
2. ResNet50 - ResNet-50模型
3. DenseNet121 - DenseNet-121模型

参考文献:
- EfficientNet: Tan, M., & Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. In ICML.
- ResNet: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In CVPR.
- DenseNet: Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In CVPR.
"""

import torch
import torch.nn as nn
import timm


class EfficientNetB4Model(nn.Module):
    """
    EfficientNet-B4模型
    
    架构:
        - Backbone: EfficientNet-B4
        - Classifier: Dropout + Linear
    
    参数:
        num_classes (int): 分类类别数，默认7
        pretrained (bool): 是否使用预训练权重，默认True
        dropout (float): Dropout比率，默认0.3
    """
    
    def __init__(self, num_classes=7, pretrained=True, dropout=0.3):
        super().__init__()
        
        # 加载EfficientNet-B4 backbone
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0  # 移除原始分类头
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'EfficientNet_B4',
            'backbone': 'efficientnet_b4',
            'num_features': self.backbone.num_features,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class ResNet50Model(nn.Module):
    """
    ResNet-50模型
    
    架构:
        - Backbone: ResNet-50
        - Classifier: Dropout + Linear
    
    参数:
        num_classes (int): 分类类别数，默认7
        pretrained (bool): 是否使用预训练权重，默认True
        dropout (float): Dropout比率，默认0.3
    """
    
    def __init__(self, num_classes=7, pretrained=True, dropout=0.3):
        super().__init__()
        
        # 加载ResNet-50 backbone
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0  # 移除原始分类头
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'ResNet50',
            'backbone': 'resnet50',
            'num_features': self.backbone.num_features,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class DenseNet121Model(nn.Module):
    """
    DenseNet-121模型
    
    架构:
        - Backbone: DenseNet-121
        - Classifier: Dropout + Linear
    
    参数:
        num_classes (int): 分类类别数，默认7
        pretrained (bool): 是否使用预训练权重，默认True
        dropout (float): Dropout比率，默认0.3
    """
    
    def __init__(self, num_classes=7, pretrained=True, dropout=0.3):
        super().__init__()
        
        # 加载DenseNet-121 backbone
        self.backbone = timm.create_model(
            'densenet121',
            pretrained=pretrained,
            num_classes=0  # 移除原始分类头
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'DenseNet121',
            'backbone': 'densenet121',
            'num_features': self.backbone.num_features,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class ResNet152Model(nn.Module):
    """
    ResNet-152模型（深度残差网络）
    
    架构:
        - Backbone: ResNet-152
        - Classifier: Dropout + Linear
    
    参数:
        num_classes (int): 分类类别数，默认7
        pretrained (bool): 是否使用预训练权重，默认True
        dropout (float): Dropout比率，默认0.3
    """
    
    def __init__(self, num_classes=7, pretrained=True, dropout=0.3):
        super().__init__()
        
        # 加载ResNet-152 backbone
        self.backbone = timm.create_model(
            'resnet152',
            pretrained=pretrained,
            num_classes=0
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'ResNet152',
            'backbone': 'resnet152',
            'num_features': self.backbone.num_features,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# 便捷函数
def get_cnn_model(model_type='efficientnet_b4', **kwargs):
    """
    获取CNN模型实例
    
    参数:
        model_type (str): 模型类型，可选'efficientnet_b4', 'resnet50', 'densenet121', 'resnet152'
        **kwargs: 传递给模型的参数
    
    返回:
        model: CNN模型实例
    """
    if model_type == 'efficientnet_b4':
        return EfficientNetB4Model(**kwargs)
    elif model_type == 'resnet50':
        return ResNet50Model(**kwargs)
    elif model_type == 'densenet121':
        return DenseNet121Model(**kwargs)
    elif model_type == 'resnet152':
        return ResNet152Model(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 示例用法
if __name__ == "__main__":
    # 测试EfficientNet_B4
    print("=" * 50)
    print("EfficientNet_B4模型")
    print("=" * 50)
    model = EfficientNetB4Model(num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试ResNet50
    print("\n" + "=" * 50)
    print("ResNet50模型")
    print("=" * 50)
    model = ResNet50Model(num_classes=7, pretrained=False)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试DenseNet121
    print("\n" + "=" * 50)
    print("DenseNet121模型")
    print("=" * 50)
    model = DenseNet121Model(num_classes=7, pretrained=False)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试ResNet152
    print("\n" + "=" * 50)
    print("ResNet152模型")
    print("=" * 50)
    model = ResNet152Model(num_classes=7, pretrained=False)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型信息: {model.get_model_info()}")

