"""
六组模型对比实验配置文件
包含5个基础模型 + Swin+Focal Loss改进模型
"""

# 六个模型配置
SIX_MODELS = [
    'ViT_Base',
    'EfficientNet_B4',
    'ResNet50',
    'DenseNet121',
    'Swin_Base',
    'Swin_Dual_Branch'  # 新增：Swin + Focal + 双分支（我们的模型）
]

# 模型显示名称
MODEL_DISPLAY_NAMES = {
    'ViT_Base': 'ViT Base',
    'EfficientNet_B4': 'EfficientNet-B4',
    'ResNet50': 'ResNet-50',
    'DenseNet121': 'DenseNet-121',
    'Swin_Base': 'Swin Base',
    'Swin_Dual_Branch': 'Swin+Focal+Dual'  # 我们的模型
}

# 模型权重路径映射（相对于models_dir的子目录）
MODEL_WEIGHT_PATHS = {
    'ViT_Base': 'ViT_Base',
    'EfficientNet_B4': 'EfficientNet_B4',
    'ResNet50': 'ResNet50',
    'DenseNet121': 'DenseNet121',
    'Swin_Base': 'Swin_Base',
    'Swin_Dual_Branch': 'swin_dual_branch'  # 双分支模型路径
}

# 模型权重文件名模板
MODEL_WEIGHT_TEMPLATES = {
    'ViT_Base': 'ViT_Base_{dataset}_best.pth',
    'EfficientNet_B4': 'EfficientNet_B4_{dataset}_best.pth',
    'ResNet50': 'ResNet50_{dataset}_best.pth',
    'DenseNet121': 'DenseNet121_{dataset}_best.pth',
    'Swin_Base': 'Swin_Base_{dataset}_best.pth',
    'Swin_Dual_Branch': '{dataset}_swin_dual_simple_best.pth'  # 双分支模型命名格式
}

# 数据集配置
DATASETS = ['BCN20000', 'HAM10000']

# 类别配置
CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
CLASS_NAMES_FULL = {
    'AKIEC': 'Actinic Keratosis',
    'BCC': 'Basal Cell Carcinoma',
    'BKL': 'Benign Keratosis',
    'DF': 'Dermatofibroma',
    'MEL': 'Melanoma',
    'NV': 'Nevus',
    'VASC': 'Vascular Lesion'
}

# 颜色配置（6个模型）
MODEL_COLORS = {
    'ViT_Base': '#FF6B6B',           # 红色
    'EfficientNet_B4': '#4ECDC4',    # 青色
    'ResNet50': '#45B7D1',           # 蓝色
    'DenseNet121': '#FFA07A',        # 浅橙色
    'Swin_Base': '#98D8C8',          # 薄荷绿
    'Swin_Dual_Branch': '#FFD93D'    # 金黄色 - 突出我们的模型
}

# 模型标记样式（用于折线图）
MODEL_MARKERS = {
    'ViT_Base': 'o',           # 圆形
    'EfficientNet_B4': 's',    # 方形
    'ResNet50': '^',           # 三角形
    'DenseNet121': 'D',        # 菱形
    'Swin_Base': 'v',          # 倒三角
    'Swin_Dual_Branch': '*'    # 星形 - 突出我们的模型
}

# 图表布局配置
LAYOUT_CONFIG = {
    'confusion_matrix': {
        'rows': 2,
        'cols': 3,
        'figsize': (18, 12)
    },
    'gradcam': {
        'rows': 7,  # 7个类别
        'cols': 7,  # 1个原图 + 6个模型
        'figsize': (28, 28)
    },
    'qualitative': {
        'rows': 4,  # 4个样本
        'cols': 7,  # 1个原图 + 6个模型
        'figsize': (28, 16)
    }
}

def get_model_weight_path(model_name, dataset, models_dir):
    """
    获取模型权重文件的完整路径
    
    Args:
        model_name: 模型名称
        dataset: 数据集名称
        models_dir: 模型目录路径
    
    Returns:
        权重文件的完整路径
    """
    from pathlib import Path
    
    weight_dir = MODEL_WEIGHT_PATHS[model_name]
    weight_template = MODEL_WEIGHT_TEMPLATES[model_name]
    weight_filename = weight_template.format(dataset=dataset)
    
    return Path(models_dir) / weight_dir / weight_filename

