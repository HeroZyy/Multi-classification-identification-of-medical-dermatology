# 数据集下载说明总结

## ✅ 已完成的工作

### 📄 更新的文件

所有主要文档都已添加数据集下载说明：

1. **README.md** (项目根目录)
   - ✅ 添加数据集下载链接
   - ✅ 说明数据集放置位置
   - ✅ 目录结构说明

2. **README_EN.md** (linux_sub/app/)
   - ✅ 详细的数据集下载说明
   - ✅ 数据集统计信息
   - ✅ 元数据格式说明

3. **README.md** (linux_sub/app/)
   - ✅ 中文版数据集下载说明
   - ✅ 类别详细说明
   - ✅ 数据集统计表格

4. **MODEL_DOWNLOAD_GUIDE.md**
   - ✅ 完整的数据集下载指南
   - ✅ 数据集使用示例
   - ✅ 双语版本

5. **QUICK_START.md**
   - ✅ 快速开始中添加数据集下载
   - ✅ 目录结构说明

6. **.gitignore**
   - ✅ 排除数据集文件夹
   - ✅ 保留评估结果文件

---

## 📊 数据集信息

### 🔗 下载链接

**Google Drive**: [https://drive.google.com/drive/folders/1oT9YuW5HMMYZdw5kzt8hj1aeVMR4Cm8q](https://drive.google.com/drive/folders/1oT9YuW5HMMYZdw5kzt8hj1aeVMR4Cm8q)

### 📋 可用数据集

| 数据集 | 图像数量 | 类别数 | 文件大小 | 说明 |
|--------|---------|--------|---------|------|
| **BCN20000** | 19,424 | 7 | ~2.5GB | 巴塞罗那医院临床数据集 |
| **HAM10000** | 10,015 | 7 | ~1.8GB | 人类vs机器数据集 |

### 📁 安装位置

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

---

## 🎯 类别说明

### 7种皮肤病变类型

| 代码 | 英文名称 | 中文名称 | 类型 | BCN数量 | HAM数量 |
|------|---------|---------|------|---------|---------|
| **NV** | Melanocytic Nevi | 良性痣 | 良性 | 12,875 | 6,705 |
| **MEL** | Melanoma | 黑色素瘤 | ⚠️ 恶性 | 3,323 | 1,113 |
| **BKL** | Benign Keratosis | 良性角化病 | 良性 | 2,624 | 1,099 |
| **BCC** | Basal Cell Carcinoma | 基底细胞癌 | 恶性 | 514 | 514 |
| **AKIEC** | Actinic Keratoses | 光化性角化病 | 癌前病变 | 67 | 327 |
| **VASC** | Vascular Lesions | 血管病变 | 良性 | 15 | 142 |
| **DF** | Dermatofibroma | 皮肤纤维瘤 | 良性 | 6 | 115 |

### 类别不平衡

- **BCN20000**: 最大类别(NV) vs 最小类别(DF) = 2146:1
- **HAM10000**: 最大类别(NV) vs 最小类别(DF) = 58:1

这就是为什么我们使用Focal Loss来处理类别不平衡问题！

---

## 📋 元数据格式

### metadata.csv 结构

```csv
image_id,diagnosis,age,sex,localization
ISIC_0000001,NV,45,male,back
ISIC_0000002,MEL,60,female,face
ISIC_0000003,BKL,55,female,chest
```

### 字段说明

- **image_id**: 图像文件名（不含.jpg扩展名）
- **diagnosis**: 诊断类别（7种之一）
- **age**: 患者年龄
- **sex**: 性别（male/female）
- **localization**: 病变位置（back, face, chest等）

---

## 💻 使用示例

### 加载数据集

```python
import pandas as pd
from PIL import Image
import os

# 加载BCN20000元数据
metadata = pd.read_csv('linux_sub/app/datasets/BCN20000/metadata.csv')

# 查看数据集信息
print(f"总图像数: {len(metadata)}")
print(f"\n类别分布:")
print(metadata['diagnosis'].value_counts())

# 加载单张图像
image_id = metadata.iloc[0]['image_id']
image_path = f'linux_sub/app/datasets/BCN20000/images/{image_id}.jpg'
image = Image.open(image_path)
print(f"\n图像尺寸: {image.size}")
```

### 数据集统计

```python
# 类别分布
class_dist = metadata['diagnosis'].value_counts()
print("类别分布:")
for cls, count in class_dist.items():
    percentage = (count / len(metadata)) * 100
    print(f"{cls}: {count} ({percentage:.2f}%)")

# 年龄分布
print(f"\n年龄范围: {metadata['age'].min()} - {metadata['age'].max()}")
print(f"平均年龄: {metadata['age'].mean():.1f}")

# 性别分布
print(f"\n性别分布:")
print(metadata['sex'].value_counts())
```

---

## 🚀 快速安装

### 一键安装脚本

```bash
# 创建数据集目录
mkdir -p linux_sub/app/datasets

# 下载并解压（手动从Google Drive下载后）
# 假设下载到了Downloads文件夹
unzip ~/Downloads/BCN20000.zip -d linux_sub/app/datasets/
unzip ~/Downloads/HAM10000.zip -d linux_sub/app/datasets/

# 验证安装
ls -la linux_sub/app/datasets/BCN20000/
ls -la linux_sub/app/datasets/HAM10000/
```

### Windows PowerShell

```powershell
# 创建数据集目录
New-Item -ItemType Directory -Force -Path "linux_sub\app\datasets"

# 解压（使用Windows内置解压或7-Zip）
Expand-Archive -Path "$env:USERPROFILE\Downloads\BCN20000.zip" -DestinationPath "linux_sub\app\datasets\"
Expand-Archive -Path "$env:USERPROFILE\Downloads\HAM10000.zip" -DestinationPath "linux_sub\app\datasets\"

# 验证安装
Get-ChildItem -Path "linux_sub\app\datasets\BCN20000\"
Get-ChildItem -Path "linux_sub\app\datasets\HAM10000\"
```

---

## 📖 相关文档

用户可以在以下位置找到数据集下载说明：

1. **主README** → 快速开始部分
2. **MODEL_DOWNLOAD_GUIDE.md** → 数据集下载部分
3. **QUICK_START.md** → 步骤4
4. **README_EN.md** → Dataset Preparation部分
5. **README.md (中文)** → 数据集下载部分

---

## ⚠️ 重要提示

### 数据集不会推送到GitHub

数据集文件已在 `.gitignore` 中排除，因为：
- ✅ 文件太大（总计~4.3GB）
- ✅ GitHub有文件大小限制
- ✅ 使用Google Drive更适合大文件分享

### 仅用于研究和教育

这些数据集仅供：
- ✅ 学术研究
- ✅ 教育目的
- ✅ 非商业用途

请遵守数据集的原始许可协议。

---

## 📞 技术支持

如有问题，请联系：
- **Email**: a1048666899@gmail.com
- **GitHub Issues**: [提交问题](https://github.com/HeroZyy/skin-lesion-classification/issues)

---

## ✅ 总结

所有数据集相关的文档和说明已经完成：

- ✅ **下载链接**: 已在所有主要文档中添加
- ✅ **安装说明**: 详细的步骤和目录结构
- ✅ **数据格式**: 元数据和类别说明
- ✅ **使用示例**: 完整的代码示例
- ✅ **文档导航**: 清晰的文档索引和链接
- ✅ **双语支持**: 中英文完整文档

**用户现在可以轻松下载和使用数据集！** 🎉

