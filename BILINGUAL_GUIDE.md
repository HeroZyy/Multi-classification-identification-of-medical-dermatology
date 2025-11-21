# 双语文档指南 / Bilingual Documentation Guide

## 📚 文档结构 / Documentation Structure

本项目提供完整的中英文双语文档，方便不同语言用户使用。

This project provides complete bilingual documentation in Chinese and English for users of different languages.

---

## 🌐 语言切换 / Language Switching

### 主README / Main README

**位置 / Location**: 项目根目录 `README.md`

这是GitHub展示的主页面，包含：
- 双语简介
- 快速开始指南
- 性能对比表
- 文档导航链接

This is the main page displayed on GitHub, containing:
- Bilingual introduction
- Quick start guide
- Performance comparison
- Documentation navigation links

### 完整文档 / Full Documentation

#### 中文版 / Chinese Version
📖 **路径 / Path**: `linux_sub/app/README.md`

包含内容 / Contents:
- ✅ 完整实验报告
- ✅ 详细性能分析
- ✅ 消融研究结果
- ✅ 技术创新说明

**切换到英文 / Switch to English**: 点击页面右上角的 "English" 按钮

#### 英文版 / English Version
📖 **路径 / Path**: `linux_sub/app/README_EN.md`

包含内容 / Contents:
- ✅ Complete experimental report
- ✅ Detailed performance analysis
- ✅ Ablation study results
- ✅ Technical innovation description

**切换到中文 / Switch to Chinese**: Click the "中文" button in the top right corner

---

## 📊 SOTA对比文档 / SOTA Comparison Documents

### 中文版 / Chinese Version
📊 **路径 / Path**: `linux_sub/app/COMPARISON_WITH_SOTA.md`

包含内容 / Contents:
- ✅ 与SOTA方法的详细对比
- ✅ 技术创新分析
- ✅ 双分支架构说明
- ✅ 性能优势总结

**切换到英文 / Switch to English**: 点击页面右上角的 "English" 按钮

### 英文版 / English Version
📊 **路径 / Path**: `linux_sub/app/COMPARISON_WITH_SOTA_EN.md`

包含内容 / Contents:
- ✅ Detailed comparison with SOTA methods
- ✅ Technical innovation analysis
- ✅ Dual-branch architecture description
- ✅ Performance advantages summary

**切换到中文 / Switch to Chinese**: Click the "中文" button in the top right corner

---

## 🎯 如何使用 / How to Use

### 方法1: 通过主README导航 / Method 1: Navigate from Main README

1. 打开项目主页 / Open project homepage
2. 在 "Documentation" 部分选择语言 / Select language in "Documentation" section
3. 点击相应链接 / Click the corresponding link

### 方法2: 直接访问文档 / Method 2: Direct Access

**中文用户 / Chinese Users**:
```
主文档: linux_sub/app/README.md
对比文档: linux_sub/app/COMPARISON_WITH_SOTA.md
教程: linux_sub/app/COMPLETE_TUTORIAL.md
```

**English Users**:
```
Main Docs: linux_sub/app/README_EN.md
Comparison: linux_sub/app/COMPARISON_WITH_SOTA_EN.md
Tutorial: linux_sub/app/COMPLETE_TUTORIAL.md (Chinese only)
```

### 方法3: 使用页面内切换按钮 / Method 3: Use In-Page Switch Button

每个文档页面右上角都有语言切换按钮：

Each document page has a language switch button in the top right corner:

```markdown
<div align="right">
  中文 | English
</div>
```

点击即可切换到对应语言版本。

Click to switch to the corresponding language version.

---

## 📋 文档对应关系 / Document Mapping

| 中文文档 / Chinese | 英文文档 / English | 内容 / Content |
|-------------------|-------------------|---------------|
| `README.md` (根目录) | `README.md` (根目录) | 双语主页 / Bilingual homepage |
| `linux_sub/app/README.md` | `linux_sub/app/README_EN.md` | 完整文档 / Full documentation |
| `linux_sub/app/COMPARISON_WITH_SOTA.md` | `linux_sub/app/COMPARISON_WITH_SOTA_EN.md` | SOTA对比 / SOTA comparison |
| `linux_sub/app/COMPLETE_TUTORIAL.md` | - | 完整教程 (仅中文) / Tutorial (Chinese only) |

---

## 🔄 内容同步 / Content Synchronization

所有双语文档内容保持同步，包括：

All bilingual documents are synchronized, including:

- ✅ 最新性能数据 / Latest performance data
- ✅ 实验结果 / Experimental results
- ✅ 模型对比 / Model comparison
- ✅ 技术细节 / Technical details
- ✅ 联系方式 / Contact information

**数据来源 / Data Source**: `linux_sub/app/evaluation_results.csv`

---

## 🎨 切换按钮样式 / Switch Button Style

### 标准样式 / Standard Style
```markdown
<div align="right">
  <strong>中文</strong> | <a href="README_EN.md">English</a>
</div>
```

显示效果 / Display:
- 当前语言加粗显示 / Current language in bold
- 其他语言显示为链接 / Other languages as links

### 徽章样式 / Badge Style
```markdown
<a href="README.md"><img src="https://img.shields.io/badge/文档-中文版-red?style=for-the-badge"></a>
<a href="README_EN.md"><img src="https://img.shields.io/badge/Docs-English-blue?style=for-the-badge"></a>
```

显示效果 / Display:
- 彩色徽章按钮 / Colorful badge buttons
- 更醒目的视觉效果 / More prominent visual effect

---

## 📱 移动端适配 / Mobile Adaptation

所有语言切换按钮在移动端也能正常显示和使用。

All language switch buttons display and work properly on mobile devices.

---

## 🤝 贡献指南 / Contribution Guide

### 添加新文档 / Adding New Documents

如果需要添加新的双语文档：

If you need to add new bilingual documents:

1. 创建中文版文档 / Create Chinese version
2. 创建对应的英文版文档 / Create corresponding English version
3. 在两个文档顶部添加语言切换按钮 / Add language switch buttons at the top
4. 更新本指南的文档对应关系表 / Update the document mapping table in this guide

### 更新现有文档 / Updating Existing Documents

更新文档时请确保：

When updating documents, please ensure:

- ✅ 同时更新中英文版本 / Update both Chinese and English versions
- ✅ 保持内容一致性 / Maintain content consistency
- ✅ 更新数据来源 / Update data sources
- ✅ 检查链接有效性 / Check link validity

---

## 📧 反馈 / Feedback

如有文档问题或建议，请联系：

For documentation issues or suggestions, please contact:

- **Email / 邮箱**: a1048666899@gmail.com
- **GitHub Issues**: [提交Issue / Submit Issue](https://github.com/HeroZyy/skin-lesion-classification/issues)

---

<div align="center">

**感谢使用本项目！**

**Thank you for using this project!**

</div>

