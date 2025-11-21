#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Swin Transformer系列模型定义

包含:
1. Swin_Base - Swin Transformer基线模型
2. Swin_Focal - Swin + Focal Loss模型
3. Swin_Dual_Branch - Swin双分支模型（通用分支 + 黑色素瘤专用分支）

参考文献:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV.
"""

import torch
import torch.nn as nn
import timm


class SwinBaseModel(nn.Module):
    """
    Swin Transformer基线模型
    
    架构:
        - Backbone: Swin-Base (swin_base_patch4_window7_224)
        - Classifier: Dropout + Linear
    
    参数:
        num_classes (int): 分类类别数，默认7
        pretrained (bool): 是否使用预训练权重，默认True
        dropout (float): Dropout比率，默认0.5
    """
    
    def __init__(self, num_classes=7, pretrained=True, dropout=0.5):
        super().__init__()
        
        # 加载Swin Transformer backbone
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
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
            'name': 'Swin_Base',
            'backbone': 'swin_base_patch4_window7_224',
            'num_features': self.backbone.num_features,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class SwinFocalModel(nn.Module):
    """
    Swin + Focal Loss模型
    
    架构与SwinBaseModel相同，但训练时使用Focal Loss
    
    参数:
        num_classes (int): 分类类别数，默认7
        pretrained (bool): 是否使用预训练权重，默认True
        dropout (float): Dropout比率，默认0.5
    """
    
    def __init__(self, num_classes=7, pretrained=True, dropout=0.5):
        super().__init__()
        
        # 加载Swin Transformer backbone
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
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
            'name': 'Swin_Focal',
            'backbone': 'swin_base_patch4_window7_224',
            'loss': 'Focal Loss',
            'num_features': self.backbone.num_features,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class SwinDualBranchModel(nn.Module):
    """
    Swin双分支模型
    
    架构:
        - Backbone: Swin-Base
        - General Branch: 通用7分类分支
        - Melanoma Branch: 黑色素瘤专用二分类分支
        - 最终输出: 将黑色素瘤分支的输出替换通用分支的黑色素瘤类别输出
    
    参数:
        num_classes (int): 分类类别数，默认7
        pretrained (bool): 是否使用预训练权重，默认True
        general_hidden (int): 通用分支隐藏层维度，默认512
        melanoma_hidden (int): 黑色素瘤分支隐藏层维度，默认256
    """
    
    def __init__(self, num_classes=7, pretrained=True, 
                 general_hidden=512, melanoma_hidden=256):
        super().__init__()
        
        # 加载Swin Transformer backbone
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0
        )
        
        hidden_dim = self.backbone.num_features
        
        # 通用分支（7分类）
        self.general_branch = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, general_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(general_hidden, num_classes)
        )
        
        # 黑色素瘤专用分支（二分类：是/否黑色素瘤）
        self.melanoma_branch = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, melanoma_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(melanoma_hidden, 2)
        )
    
    def forward(self, x):
        """
        前向传播
        
        流程:
        1. 通过backbone提取特征
        2. 通用分支预测7个类别
        3. 黑色素瘤分支预测是否为黑色素瘤
        4. 用黑色素瘤分支的输出替换通用分支的黑色素瘤类别（索引4）
        """
        # 提取特征
        features = self.backbone(x)
        
        # 通用分支输出
        general_out = self.general_branch(features)
        
        # 黑色素瘤分支输出
        melanoma_out = self.melanoma_branch(features)
        
        # 融合输出：用黑色素瘤分支的"是黑色素瘤"概率替换通用分支的黑色素瘤类别
        output = general_out.clone()
        output[:, 4] = melanoma_out[:, 0]  # 索引4是黑色素瘤类别
        
        return output
    
    def get_branch_outputs(self, x):
        """
        获取两个分支的独立输出（用于分析）
        
        返回:
            general_out: 通用分支输出
            melanoma_out: 黑色素瘤分支输出
        """
        features = self.backbone(x)
        general_out = self.general_branch(features)
        melanoma_out = self.melanoma_branch(features)
        return general_out, melanoma_out
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'name': 'Swin_Dual_Branch',
            'backbone': 'swin_base_patch4_window7_224',
            'architecture': 'Dual Branch (General + Melanoma)',
            'general_branch': '7-class classification',
            'melanoma_branch': '2-class classification (melanoma vs non-melanoma)',
            'num_features': self.backbone.num_features,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# 便捷函数
def get_swin_model(model_type='base', **kwargs):
    """
    获取Swin模型实例
    
    参数:
        model_type (str): 模型类型，可选'base', 'focal', 'dual_branch'
        **kwargs: 传递给模型的参数
    
    返回:
        model: Swin模型实例
    """
    if model_type == 'base':
        return SwinBaseModel(**kwargs)
    elif model_type == 'focal':
        return SwinFocalModel(**kwargs)
    elif model_type == 'dual_branch':
        return SwinDualBranchModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 示例用法
if __name__ == "__main__":
    # 测试Swin_Base
    print("=" * 50)
    print("Swin_Base模型")
    print("=" * 50)
    model = SwinBaseModel(num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试Swin_Focal
    print("\n" + "=" * 50)
    print("Swin_Focal模型")
    print("=" * 50)
    model = SwinFocalModel(num_classes=7, pretrained=False)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试Swin_Dual_Branch
    print("\n" + "=" * 50)
    print("Swin_Dual_Branch模型")
    print("=" * 50)
    model = SwinDualBranchModel(num_classes=7, pretrained=False)
    output = model(x)
    general_out, melanoma_out = model.get_branch_outputs(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"通用分支输出形状: {general_out.shape}")
    print(f"黑色素瘤分支输出形状: {melanoma_out.shape}")
    print(f"模型信息: {model.get_model_info()}")

