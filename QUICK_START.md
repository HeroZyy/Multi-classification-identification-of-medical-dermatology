# 快速开始指南 / Quick Start Guide

<div align="right">
  <strong>中文</strong> | <a href="#english-version">English</a>
</div>

---

## 中文版

### 🚀 5分钟快速开始

本指南将帮助您在5分钟内开始使用我们的皮肤病变分类模型。

### 📋 前置要求

- Python 3.8+
- CUDA 11.0+ (如果使用GPU)
- 至少8GB RAM
- 至少4GB GPU显存 (推荐)

### ⚡ 快速安装

#### 步骤1: 克隆仓库

```bash
git clone https://github.com/HeroZyy/skin-lesion-classification.git
cd skin-lesion-classification
```

#### 步骤2: 创建环境

```bash
# 使用conda (推荐)
conda create -n skin_lesion python=3.8
conda activate skin_lesion

# 或使用venv
python -m venv skin_lesion_env
source skin_lesion_env/bin/activate  # Linux/Mac
# skin_lesion_env\Scripts\activate  # Windows
```

#### 步骤3: 安装依赖

```bash
# 安装PyTorch (根据您的CUDA版本选择)
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
# pip install torch torchvision

# 安装其他依赖
pip install timm scikit-learn pandas numpy matplotlib seaborn pillow opencv-python tqdm
```

#### 步骤4: 下载预训练模型和数据集

1. 访问 [Google Drive](https://drive.google.com/drive/folders/1oT9YuW5HMMYZdw5kzt8hj1aeVMR4Cm8q)
2. 下载以下文件：
   - **预训练模型**: `HAM10000_best_model.pth` (推荐，98.90%准确率)
   - **数据集** (可选): `BCN20000.zip` 和/或 `HAM10000.zip`

3. 创建目录并放置文件：

```bash
# 创建模型目录
mkdir -p linux_sub/app/models/five_model_comparison_final/models/swin_dual_branch
# 将下载的模型文件移动到上述目录

# 创建数据集目录（如果需要训练或测试）
mkdir -p linux_sub/app/datasets
# 解压数据集到上述目录
```

**目录结构**:
```
linux_sub/app/
├── models/five_model_comparison_final/models/swin_dual_branch/
│   └── HAM10000_best_model.pth
└── datasets/
    ├── BCN20000/
    │   ├── images/
    │   └── metadata.csv
    └── HAM10000/
        ├── images/
        └── metadata.csv
```

### 🔍 快速推理

创建文件 `quick_inference.py`:

```python
import torch
from PIL import Image
from torchvision import transforms
from models.model_loader import ModelLoader

# 1. 加载模型
print("加载模型...")
loader = ModelLoader()
model = loader.load_swin_dual_model("HAM10000", best=True)
model.eval()

# 2. 加载并预处理图像
image_path = "your_image.jpg"  # 替换为您的图像路径
print(f"加载图像: {image_path}")
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)

# 3. 推理
print("进行预测...")
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    prediction = output.argmax(dim=1).item()
    confidence = probabilities.max().item()

# 4. 显示结果
class_names = ['NV', 'MEL', 'BKL', 'BCC', 'AKIEC', 'VASC', 'DF']
class_descriptions = {
    'NV': '良性痣',
    'MEL': '黑色素瘤 ⚠️',
    'BKL': '良性角化病',
    'BCC': '基底细胞癌',
    'AKIEC': '光化性角化病',
    'VASC': '血管病变',
    'DF': '皮肤纤维瘤'
}

print("\n" + "="*50)
print(f"预测结果: {class_names[prediction]} - {class_descriptions[class_names[prediction]]}")
print(f"置信度: {confidence:.2%}")
print("="*50)
print("\n所有类别概率:")
for name, prob in zip(class_names, probabilities[0]):
    bar = "█" * int(prob * 50)
    print(f"{name:6s} {prob:.2%} {bar}")
```

运行：

```bash
python quick_inference.py
```

### 📊 预期输出

```
加载模型...
加载图像: your_image.jpg
进行预测...

==================================================
预测结果: MEL - 黑色素瘤 ⚠️
置信度: 97.85%
==================================================

所有类别概率:
NV     1.23% ▌
MEL    97.85% ████████████████████████████████████████████████
BKL    0.45% 
BCC    0.32% 
AKIEC  0.08% 
VASC   0.05% 
DF     0.02% 
```

### 🎯 下一步

1. **查看完整文档**: [README.md](README.md)
2. **了解模型详情**: [MODEL_DOWNLOAD_GUIDE.md](MODEL_DOWNLOAD_GUIDE.md)
3. **学习完整教程**: [COMPLETE_TUTORIAL.md](COMPLETE_TUTORIAL.md)
4. **对比SOTA方法**: [COMPARISON_WITH_SOTA.md](COMPARISON_WITH_SOTA.md)

### 💡 提示

- **推荐使用HAM10000模型**: 准确率98.90%，性能最佳
- **GPU加速**: 使用GPU可显著提升推理速度
- **批量处理**: 可以一次处理多张图像以提高效率

### 📞 需要帮助？

- **文档**: 查看 [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)
- **问题**: 提交 [GitHub Issue](https://github.com/HeroZyy/skin-lesion-classification/issues)
- **邮件**: a1048666899@gmail.com

---

<div id="english-version"></div>

## English Version

### 🚀 5-Minute Quick Start

This guide will help you get started with our skin lesion classification model in 5 minutes.

### 📋 Prerequisites

- Python 3.8+
- CUDA 11.0+ (if using GPU)
- At least 8GB RAM
- At least 4GB GPU VRAM (recommended)

### ⚡ Quick Installation

#### Step 1: Clone Repository

```bash
git clone https://github.com/HeroZyy/skin-lesion-classification.git
cd skin-lesion-classification
```

#### Step 2: Create Environment

```bash
# Using conda (recommended)
conda create -n skin_lesion python=3.8
conda activate skin_lesion

# Or using venv
python -m venv skin_lesion_env
source skin_lesion_env/bin/activate  # Linux/Mac
# skin_lesion_env\Scripts\activate  # Windows
```

#### Step 3: Install Dependencies

```bash
# Install PyTorch (choose based on your CUDA version)
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
# pip install torch torchvision

# Install other dependencies
pip install timm scikit-learn pandas numpy matplotlib seaborn pillow opencv-python tqdm
```

#### Step 4: Download Pre-trained Models and Datasets

1. Visit [Google Drive](https://drive.google.com/drive/folders/1oT9YuW5HMMYZdw5kzt8hj1aeVMR4Cm8q)
2. Download the following files:
   - **Pre-trained model**: `HAM10000_best_model.pth` (recommended, 98.90% accuracy)
   - **Datasets** (optional): `BCN20000.zip` and/or `HAM10000.zip`

3. Create directories and place files:

```bash
# Create model directory
mkdir -p linux_sub/app/models/five_model_comparison_final/models/swin_dual_branch
# Move downloaded model file to the above directory

# Create datasets directory (if you need to train or test)
mkdir -p linux_sub/app/datasets
# Extract datasets to the above directory
```

**Directory Structure**:
```
linux_sub/app/
├── models/five_model_comparison_final/models/swin_dual_branch/
│   └── HAM10000_best_model.pth
└── datasets/
    ├── BCN20000/
    │   ├── images/
    │   └── metadata.csv
    └── HAM10000/
        ├── images/
        └── metadata.csv
```

### 🔍 Quick Inference

Create file `quick_inference.py`:

```python
import torch
from PIL import Image
from torchvision import transforms
from models.model_loader import ModelLoader

# 1. Load model
print("Loading model...")
loader = ModelLoader()
model = loader.load_swin_dual_model("HAM10000", best=True)
model.eval()

# 2. Load and preprocess image
image_path = "your_image.jpg"  # Replace with your image path
print(f"Loading image: {image_path}")
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)

# 3. Inference
print("Making prediction...")
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    prediction = output.argmax(dim=1).item()
    confidence = probabilities.max().item()

# 4. Display results
class_names = ['NV', 'MEL', 'BKL', 'BCC', 'AKIEC', 'VASC', 'DF']
class_descriptions = {
    'NV': 'Melanocytic Nevi (Benign)',
    'MEL': 'Melanoma (Malignant) ⚠️',
    'BKL': 'Benign Keratosis',
    'BCC': 'Basal Cell Carcinoma',
    'AKIEC': 'Actinic Keratoses',
    'VASC': 'Vascular Lesions',
    'DF': 'Dermatofibroma'
}

print("\n" + "="*50)
print(f"Prediction: {class_names[prediction]} - {class_descriptions[class_names[prediction]]}")
print(f"Confidence: {confidence:.2%}")
print("="*50)
print("\nAll Class Probabilities:")
for name, prob in zip(class_names, probabilities[0]):
    bar = "█" * int(prob * 50)
    print(f"{name:6s} {prob:.2%} {bar}")
```

Run:

```bash
python quick_inference.py
```

### 📊 Expected Output

```
Loading model...
Loading image: your_image.jpg
Making prediction...

==================================================
Prediction: MEL - Melanoma (Malignant) ⚠️
Confidence: 97.85%
==================================================

All Class Probabilities:
NV     1.23% ▌
MEL    97.85% ████████████████████████████████████████████████
BKL    0.45%
BCC    0.32%
AKIEC  0.08%
VASC   0.05%
DF     0.02%
```

### 🎯 Next Steps

1. **View Full Documentation**: [README_EN.md](README_EN.md)
2. **Learn Model Details**: [MODEL_DOWNLOAD_GUIDE.md](MODEL_DOWNLOAD_GUIDE.md)
3. **Complete Tutorial**: [COMPLETE_TUTORIAL.md](COMPLETE_TUTORIAL.md)
4. **SOTA Comparison**: [COMPARISON_WITH_SOTA_EN.md](COMPARISON_WITH_SOTA_EN.md)

### 💡 Tips

- **Recommended: Use HAM10000 model**: 98.90% accuracy, best performance
- **GPU Acceleration**: Using GPU significantly improves inference speed
- **Batch Processing**: Process multiple images at once for better efficiency

### 📞 Need Help?

- **Documentation**: See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)
- **Issues**: Submit [GitHub Issue](https://github.com/HeroZyy/skin-lesion-classification/issues)
- **Email**: a1048666899@gmail.com

---

<div align="center">

**Made with ❤️ for advancing medical AI**

**用 ❤️ 推进医学人工智能**

[⬆ Back to Top](#快速开始指南--quick-start-guide)

</div>


