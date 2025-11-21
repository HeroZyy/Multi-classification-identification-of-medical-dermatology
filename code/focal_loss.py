#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Focal Loss损失函数定义
用于处理类别不平衡问题

参考文献:
Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
Focal loss for dense object detection. In ICCV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss - 解决类别不平衡问题
    
    公式: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    参数:
        alpha (float or list): 类别权重，默认0.25
        gamma (float): 聚焦参数，默认2.0
        num_classes (int): 类别数量，默认7
        reduction (str): 损失聚合方式，'mean'或'sum'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=7, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        
        # 设置alpha权重
        if isinstance(alpha, (float, int)):
            self.alpha = torch.ones(num_classes) * alpha
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha
    
    def forward(self, inputs, targets):
        """
        前向传播
        
        参数:
            inputs: 模型输出logits，shape (batch_size, num_classes)
            targets: 真实标签，shape (batch_size,)
        
        返回:
            loss: Focal Loss值
        """
        # 计算交叉熵损失（不进行reduction）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算p_t
        pt = torch.exp(-ce_loss)
        
        # 获取对应类别的alpha值
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
        else:
            alpha_t = 1.0
        
        # 计算Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        # 聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    自适应Focal Loss - 根据类别频率自动调整alpha
    
    参数:
        gamma (float): 聚焦参数，默认2.0
        num_classes (int): 类别数量，默认7
        class_freq (list): 各类别的样本频率
    """
    
    def __init__(self, gamma=2.0, num_classes=7, class_freq=None):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        
        # 根据类别频率计算alpha
        if class_freq is not None:
            class_freq = torch.tensor(class_freq, dtype=torch.float32)
            # 使用逆频率作为权重
            self.alpha = 1.0 / (class_freq + 1e-6)
            # 归一化
            self.alpha = self.alpha / self.alpha.sum() * num_classes
        else:
            self.alpha = torch.ones(num_classes)
    
    def forward(self, inputs, targets):
        """前向传播"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        alpha_t = self.alpha.to(inputs.device)[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class MelanomaFocalLoss(nn.Module):
    """
    黑色素瘤专用Focal Loss - 为黑色素瘤类别设置更高的权重
    
    参数:
        alpha (float): 基础alpha值，默认0.25
        gamma (float): 聚焦参数，默认2.0
        melanoma_weight (float): 黑色素瘤额外权重倍数，默认2.0
        num_classes (int): 类别数量，默认7
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, melanoma_weight=2.0, num_classes=7):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        
        # 为所有类别设置基础alpha
        self.alpha = torch.ones(num_classes) * alpha
        
        # 为黑色素瘤（索引4）设置更高的权重
        self.alpha[4] = alpha * melanoma_weight
    
    def forward(self, inputs, targets):
        """前向传播"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        alpha_t = self.alpha.to(inputs.device)[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


# 便捷函数
def get_focal_loss(loss_type='standard', **kwargs):
    """
    获取Focal Loss实例
    
    参数:
        loss_type (str): 损失类型，可选'standard', 'adaptive', 'melanoma'
        **kwargs: 传递给损失函数的参数
    
    返回:
        loss_fn: Focal Loss实例
    """
    if loss_type == 'standard':
        return FocalLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveFocalLoss(**kwargs)
    elif loss_type == 'melanoma':
        return MelanomaFocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    batch_size = 32
    num_classes = 7
    
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 标准Focal Loss
    print("=" * 50)
    print("标准Focal Loss")
    print("=" * 50)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(inputs, targets)
    print(f"Loss: {loss.item():.4f}")
    
    # 自适应Focal Loss
    print("\n" + "=" * 50)
    print("自适应Focal Loss")
    print("=" * 50)
    class_freq = [0.1, 0.15, 0.2, 0.05, 0.1, 0.35, 0.05]  # 示例类别频率
    adaptive_focal_loss = AdaptiveFocalLoss(gamma=2.0, class_freq=class_freq)
    loss = adaptive_focal_loss(inputs, targets)
    print(f"Loss: {loss.item():.4f}")
    print(f"Alpha weights: {adaptive_focal_loss.alpha}")
    
    # 黑色素瘤专用Focal Loss
    print("\n" + "=" * 50)
    print("黑色素瘤专用Focal Loss")
    print("=" * 50)
    melanoma_focal_loss = MelanomaFocalLoss(alpha=0.25, gamma=2.0, melanoma_weight=2.0)
    loss = melanoma_focal_loss(inputs, targets)
    print(f"Loss: {loss.item():.4f}")
    print(f"Alpha weights: {melanoma_focal_loss.alpha}")

