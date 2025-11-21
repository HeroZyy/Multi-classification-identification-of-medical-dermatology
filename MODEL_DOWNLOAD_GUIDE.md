# 预训练模型和数据集下载指南 / Pre-trained Model & Dataset Download Guide

<div align="right">
  <strong>中文</strong> | <a href="#english-version">English</a>
</div>

---

## 中文版

### 📦 下载资源

我们提供了在BCN20000和HAM10000数据集上训练的高性能预训练模型，以及完整的数据集。

**🔗 Google Drive下载链接**: [https://drive.google.com/drive/folders/1oT9YuW5HMMYZdw5kzt8hj1aeVMR4Cm8q](https://drive.google.com/drive/folders/1oT9YuW5HMMYZdw5kzt8hj1aeVMR4Cm8q)

**包含内容**:
- ✅ 预训练模型 (4个.pth文件)
- ✅ BCN20000数据集 (19,424张图像)
- ✅ HAM10000数据集 (10,015张图像)

---

## 📊 数据集下载

### 可用数据集

| 数据集 | 图像数量 | 类别数 | 文件大小 | 说明 |
|--------|---------|--------|---------|------|
| **BCN20000** | 19,424 | 7 | ~2.5GB | 巴塞罗那医院数据集 |
| **HAM10000** | 10,015 | 7 | ~1.8GB | 人类vs机器数据集 |

### 🚀 数据集安装步骤

#### 步骤1: 下载数据集

1. 访问Google Drive链接
2. 下载 `BCN20000.zip` 和 `HAM10000.zip`
3. 保存到本地

#### 步骤2: 创建目录结构

在项目根目录下创建数据集目录：

```bash
mkdir -p linux_sub/app/datasets
```

#### 步骤3: 解压数据集

将下载的数据集解压到 `linux_sub/app/datasets/` 目录：

```
linux_sub/app/datasets/
├── BCN20000/
│   ├── images/
│   │   ├── ISIC_0000001.jpg
│   │   ├── ISIC_0000002.jpg
│   │   └── ... (19,424张图像)
│   └── metadata.csv
└── HAM10000/
    ├── images/
    │   ├── ISIC_0024306.jpg
    │   ├── ISIC_0024307.jpg
    │   └── ... (10,015张图像)
    └── metadata.csv
```

### 📋 数据集格式

#### 元数据文件 (metadata.csv)

```csv
image_id,diagnosis,age,sex,localization
ISIC_0000001,NV,45,male,back
ISIC_0000002,MEL,60,female,face
ISIC_0000003,BKL,55,female,chest
```

**字段说明**:
- `image_id`: 图像文件名（不含扩展名）
- `diagnosis`: 诊断类别（NV, MEL, BKL, BCC, AKIEC, VASC, DF）
- `age`: 患者年龄
- `sex`: 性别（male/female）
- `localization`: 病变位置

#### 类别说明

| 代码 | 英文名称 | 中文名称 | 类型 | BCN数量 | HAM数量 |
|------|---------|---------|------|---------|---------|
| **NV** | Melanocytic Nevi | 良性痣 | 良性 | 12,875 | 6,705 |
| **MEL** | Melanoma | 黑色素瘤 | ⚠️ 恶性 | 3,323 | 1,113 |
| **BKL** | Benign Keratosis | 良性角化病 | 良性 | 2,624 | 1,099 |
| **BCC** | Basal Cell Carcinoma | 基底细胞癌 | 恶性 | 514 | 514 |
| **AKIEC** | Actinic Keratoses | 光化性角化病 | 癌前 | 67 | 327 |
| **VASC** | Vascular Lesions | 血管病变 | 良性 | 15 | 142 |
| **DF** | Dermatofibroma | 皮肤纤维瘤 | 良性 | 6 | 115 |

### 💻 使用数据集

#### 加载数据集示例

```python
import pandas as pd
from PIL import Image
import os

# 加载元数据
metadata = pd.read_csv('linux_sub/app/datasets/BCN20000/metadata.csv')

# 查看数据集信息
print(f"总图像数: {len(metadata)}")
print(f"类别分布:\n{metadata['diagnosis'].value_counts()}")

# 加载单张图像
image_id = metadata.iloc[0]['image_id']
image_path = f'linux_sub/app/datasets/BCN20000/images/{image_id}.jpg'
image = Image.open(image_path)
print(f"图像尺寸: {image.size}")
```

---

## 🔧 预训练模型下载

### 📋 可用模型

| 模型文件 | 数据集 | 准确率 | MEL F1 | 文件大小 | 说明 |
|---------|--------|--------|--------|---------|------|
| `BCN20000_best_model.pth` | BCN20000 | 93.33% | 0.974 | ~350MB | 最佳验证性能 |
| `HAM10000_best_model.pth` | HAM10000 | **98.90%** | **0.977** | ~350MB | **推荐使用** ⭐ |
| `BCN20000_final_model.pth` | BCN20000 | 93.33% | 0.974 | ~350MB | 最终训练模型 |
| `HAM10000_final_model.pth` | HAM10000 | 98.90% | 0.977 | ~350MB | 最终训练模型 |

### 🚀 安装步骤

#### 步骤1: 下载模型

1. 访问Google Drive链接
2. 选择需要的模型文件
3. 点击下载（或添加到您的Google Drive后下载）

#### 步骤2: 创建目录结构

在项目根目录下创建以下目录结构：

```bash
mkdir -p linux_sub/app/models/five_model_comparison_final/models/swin_dual_branch
```

#### 步骤3: 放置模型文件

将下载的模型文件放置到以下目录：

```
linux_sub/app/models/five_model_comparison_final/models/swin_dual_branch/
├── BCN20000_best_model.pth
├── HAM10000_best_model.pth
├── BCN20000_final_model.pth
└── HAM10000_final_model.pth
```

### 💻 使用方法

#### 方法1: 使用ModelLoader（推荐）

```python
from models.model_loader import ModelLoader

# 初始化加载器
loader = ModelLoader()

# 加载HAM10000最佳模型（推荐）
model = loader.load_swin_dual_model("HAM10000", best=True)

# 或加载BCN20000最佳模型
# model = loader.load_swin_dual_model("BCN20000", best=True)

# 设置为评估模式
model.eval()
```

#### 方法2: 直接加载

```python
import torch
from models.swin_dual_branch import SwinDualBranchAttentionModel

# 创建模型
model = SwinDualBranchAttentionModel(num_classes=7)

# 加载权重
checkpoint = torch.load(
    'linux_sub/app/models/five_model_comparison_final/models/swin_dual_branch/HAM10000_best_model.pth',
    map_location='cpu'
)
model.load_state_dict(checkpoint['model_state_dict'])

# 设置为评估模式
model.eval()
```

### 🔍 完整推理示例

```python
import torch
from PIL import Image
from torchvision import transforms
from models.model_loader import ModelLoader

# 1. 加载模型
loader = ModelLoader()
model = loader.load_swin_dual_model("HAM10000", best=True)
model.eval()

# 2. 准备图像
image_path = "path/to/your/skin_lesion_image.jpg"
image = Image.open(image_path).convert('RGB')

# 3. 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
image_tensor = transform(image).unsqueeze(0)

# 4. 推理
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    prediction = output.argmax(dim=1).item()
    confidence = probabilities.max().item()

# 5. 解析结果
class_names = ['NV', 'MEL', 'BKL', 'BCC', 'AKIEC', 'VASC', 'DF']
class_descriptions = {
    'NV': '良性痣 (Melanocytic Nevi)',
    'MEL': '黑色素瘤 (Melanoma)',
    'BKL': '良性角化病 (Benign Keratosis)',
    'BCC': '基底细胞癌 (Basal Cell Carcinoma)',
    'AKIEC': '光化性角化病 (Actinic Keratoses)',
    'VASC': '血管病变 (Vascular Lesions)',
    'DF': '皮肤纤维瘤 (Dermatofibroma)'
}

predicted_class = class_names[prediction]
print(f"预测类别: {predicted_class}")
print(f"类别描述: {class_descriptions[predicted_class]}")
print(f"置信度: {confidence:.2%}")
print(f"\n所有类别概率:")
for i, (name, prob) in enumerate(zip(class_names, probabilities[0])):
    print(f"  {name}: {prob:.2%}")
```

### 📊 模型性能

#### HAM10000数据集（推荐）

| 指标 | 数值 |
|------|------|
| 总体准确率 | **98.90%** 🏆 |
| Macro F1 | 0.984 |
| Weighted F1 | 0.989 |
| 黑色素瘤F1 | **0.977** |

#### BCN20000数据集

| 指标 | 数值 |
|------|------|
| 总体准确率 | 93.33% |
| Macro F1 | 0.930 |
| Weighted F1 | 0.936 |
| 黑色素瘤F1 | 0.974 |

### ⚙️ 系统要求

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **GPU**: 推荐使用GPU（至少4GB显存）
- **CPU**: 也可使用CPU推理（速度较慢）
- **内存**: 至少8GB RAM

### 🔧 故障排除

#### 问题1: 找不到模型文件

**错误信息**: `FileNotFoundError: [Errno 2] No such file or directory`

**解决方案**:
- 确认模型文件已下载
- 检查文件路径是否正确
- 确保目录结构与上述一致

#### 问题2: CUDA内存不足

**错误信息**: `RuntimeError: CUDA out of memory`

**解决方案**:
```python
# 使用CPU推理
model = model.to('cpu')
image_tensor = image_tensor.to('cpu')

# 或减小batch size
```

#### 问题3: 模型加载失败

**错误信息**: `RuntimeError: Error(s) in loading state_dict`

**解决方案**:
- 确认下载的模型文件完整（未损坏）
- 检查PyTorch版本兼容性
- 重新下载模型文件

### 📞 技术支持

如有问题，请联系：
- **Email**: a1048666899@gmail.com
- **GitHub Issues**: [提交问题](https://github.com/HeroZyy/skin-lesion-classification/issues)

---

<div id="english-version"></div>

## English Version

### 📦 Download Resources

We provide high-performance pre-trained models trained on BCN20000 and HAM10000 datasets, along with the complete datasets.

**🔗 Google Drive Download Link**: [https://drive.google.com/drive/folders/1oT9YuW5HMMYZdw5kzt8hj1aeVMR4Cm8q](https://drive.google.com/drive/folders/1oT9YuW5HMMYZdw5kzt8hj1aeVMR4Cm8q)

**Contents**:
- ✅ Pre-trained models (4 .pth files)
- ✅ BCN20000 dataset (19,424 images)
- ✅ HAM10000 dataset (10,015 images)

---

## 📊 Dataset Download

### Available Datasets

| Dataset | Images | Classes | File Size | Description |
|---------|--------|---------|-----------|-------------|
| **BCN20000** | 19,424 | 7 | ~2.5GB | Barcelona Hospital Clinic dataset |
| **HAM10000** | 10,015 | 7 | ~1.8GB | Human Against Machine dataset |

### 🚀 Dataset Installation Steps

#### Step 1: Download Datasets

1. Visit the Google Drive link
2. Download `BCN20000.zip` and `HAM10000.zip`
3. Save to local storage

#### Step 2: Create Directory Structure

Create the datasets directory in your project root:

```bash
mkdir -p linux_sub/app/datasets
```

#### Step 3: Extract Datasets

Extract the downloaded datasets to `linux_sub/app/datasets/`:

```
linux_sub/app/datasets/
├── BCN20000/
│   ├── images/
│   │   ├── ISIC_0000001.jpg
│   │   ├── ISIC_0000002.jpg
│   │   └── ... (19,424 images)
│   └── metadata.csv
└── HAM10000/
    ├── images/
    │   ├── ISIC_0024306.jpg
    │   ├── ISIC_0024307.jpg
    │   └── ... (10,015 images)
    └── metadata.csv
```

### 📋 Dataset Format

#### Metadata File (metadata.csv)

```csv
image_id,diagnosis,age,sex,localization
ISIC_0000001,NV,45,male,back
ISIC_0000002,MEL,60,female,face
ISIC_0000003,BKL,55,female,chest
```

**Field Descriptions**:
- `image_id`: Image filename (without extension)
- `diagnosis`: Diagnosis class (NV, MEL, BKL, BCC, AKIEC, VASC, DF)
- `age`: Patient age
- `sex`: Gender (male/female)
- `localization`: Lesion location

#### Class Descriptions

| Code | English Name | Type | BCN Count | HAM Count |
|------|-------------|------|-----------|-----------|
| **NV** | Melanocytic Nevi | Benign | 12,875 | 6,705 |
| **MEL** | Melanoma | ⚠️ Malignant | 3,323 | 1,113 |
| **BKL** | Benign Keratosis | Benign | 2,624 | 1,099 |
| **BCC** | Basal Cell Carcinoma | Malignant | 514 | 514 |
| **AKIEC** | Actinic Keratoses | Pre-cancerous | 67 | 327 |
| **VASC** | Vascular Lesions | Benign | 15 | 142 |
| **DF** | Dermatofibroma | Benign | 6 | 115 |

### 💻 Using the Dataset

#### Loading Dataset Example

```python
import pandas as pd
from PIL import Image
import os

# Load metadata
metadata = pd.read_csv('linux_sub/app/datasets/BCN20000/metadata.csv')

# View dataset info
print(f"Total images: {len(metadata)}")
print(f"Class distribution:\n{metadata['diagnosis'].value_counts()}")

# Load a single image
image_id = metadata.iloc[0]['image_id']
image_path = f'linux_sub/app/datasets/BCN20000/images/{image_id}.jpg'
image = Image.open(image_path)
print(f"Image size: {image.size}")
```

---

## 🔧 Pre-trained Model Download

### 📋 Available Models

| Model File | Dataset | Accuracy | MEL F1 | File Size | Description |
|-----------|---------|----------|--------|-----------|-------------|
| `BCN20000_best_model.pth` | BCN20000 | 93.33% | 0.974 | ~350MB | Best validation performance |
| `HAM10000_best_model.pth` | HAM10000 | **98.90%** | **0.977** | ~350MB | **Recommended** ⭐ |
| `BCN20000_final_model.pth` | BCN20000 | 93.33% | 0.974 | ~350MB | Final trained model |
| `HAM10000_final_model.pth` | HAM10000 | 98.90% | 0.977 | ~350MB | Final trained model |

### 🚀 Installation Steps

#### Step 1: Download Models

1. Visit the Google Drive link
2. Select the model files you need
3. Click download (or add to your Google Drive then download)

#### Step 2: Create Directory Structure

Create the following directory structure in your project root:

```bash
mkdir -p linux_sub/app/models/five_model_comparison_final/models/swin_dual_branch
```

#### Step 3: Place Model Files

Place the downloaded model files in the following directory:

```
linux_sub/app/models/five_model_comparison_final/models/swin_dual_branch/
├── BCN20000_best_model.pth
├── HAM10000_best_model.pth
├── BCN20000_final_model.pth
└── HAM10000_final_model.pth
```

### 💻 Usage

#### Method 1: Using ModelLoader (Recommended)

```python
from models.model_loader import ModelLoader

# Initialize loader
loader = ModelLoader()

# Load HAM10000 best model (recommended)
model = loader.load_swin_dual_model("HAM10000", best=True)

# Or load BCN20000 best model
# model = loader.load_swin_dual_model("BCN20000", best=True)

# Set to evaluation mode
model.eval()
```

#### Method 2: Direct Loading

```python
import torch
from models.swin_dual_branch import SwinDualBranchAttentionModel

# Create model
model = SwinDualBranchAttentionModel(num_classes=7)

# Load weights
checkpoint = torch.load(
    'linux_sub/app/models/five_model_comparison_final/models/swin_dual_branch/HAM10000_best_model.pth',
    map_location='cpu'
)
model.load_state_dict(checkpoint['model_state_dict'])

# Set to evaluation mode
model.eval()
```

### 🔍 Complete Inference Example

```python
import torch
from PIL import Image
from torchvision import transforms
from models.model_loader import ModelLoader

# 1. Load model
loader = ModelLoader()
model = loader.load_swin_dual_model("HAM10000", best=True)
model.eval()

# 2. Prepare image
image_path = "path/to/your/skin_lesion_image.jpg"
image = Image.open(image_path).convert('RGB')

# 3. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
image_tensor = transform(image).unsqueeze(0)

# 4. Inference
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    prediction = output.argmax(dim=1).item()
    confidence = probabilities.max().item()

# 5. Parse results
class_names = ['NV', 'MEL', 'BKL', 'BCC', 'AKIEC', 'VASC', 'DF']
class_descriptions = {
    'NV': 'Melanocytic Nevi (Benign Mole)',
    'MEL': 'Melanoma (Malignant)',
    'BKL': 'Benign Keratosis',
    'BCC': 'Basal Cell Carcinoma',
    'AKIEC': 'Actinic Keratoses',
    'VASC': 'Vascular Lesions',
    'DF': 'Dermatofibroma'
}

predicted_class = class_names[prediction]
print(f"Predicted Class: {predicted_class}")
print(f"Description: {class_descriptions[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
print(f"\nAll Class Probabilities:")
for i, (name, prob) in enumerate(zip(class_names, probabilities[0])):
    print(f"  {name}: {prob:.2%}")
```

### 📊 Model Performance

#### HAM10000 Dataset (Recommended)

| Metric | Value |
|--------|-------|
| Overall Accuracy | **98.90%** 🏆 |
| Macro F1 | 0.984 |
| Weighted F1 | 0.989 |
| Melanoma F1 | **0.977** |

#### BCN20000 Dataset

| Metric | Value |
|--------|-------|
| Overall Accuracy | 93.33% |
| Macro F1 | 0.930 |
| Weighted F1 | 0.936 |
| Melanoma F1 | 0.974 |

### ⚙️ System Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **GPU**: Recommended (at least 4GB VRAM)
- **CPU**: Can also use CPU inference (slower)
- **RAM**: At least 8GB

### 🔧 Troubleshooting

#### Issue 1: Model File Not Found

**Error Message**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**:
- Confirm model files are downloaded
- Check file path is correct
- Ensure directory structure matches above

#### Issue 2: CUDA Out of Memory

**Error Message**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Use CPU inference
model = model.to('cpu')
image_tensor = image_tensor.to('cpu')

# Or reduce batch size
```

#### Issue 3: Model Loading Failed

**Error Message**: `RuntimeError: Error(s) in loading state_dict`

**Solution**:
- Confirm downloaded model file is complete (not corrupted)
- Check PyTorch version compatibility
- Re-download model file

### 📞 Technical Support

For questions, please contact:
- **Email**: a1048666899@gmail.com
- **GitHub Issues**: [Submit Issue](https://github.com/HeroZyy/skin-lesion-classification/issues)

---

<div align="center">

**Made with ❤️ for advancing medical AI**

**用 ❤️ 推进医学人工智能**

[⬆ Back to Top](#预训练模型下载指南--pre-trained-model-download-guide)

</div>


