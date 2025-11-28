下面是**完整中文翻译**，并且**严格保持你给的原始格式、HTML、Markdown、注释、缩进完全一致**：

---

<div align="center">
  <h1><img src="assets/logo.jpeg" alt="TokenCast logo" style="height: 1em; width: auto; vertical-align: -0.15em; margin-right: 0.4em;">TokenCast：一种通过符号离散化实现上下文感知时间序列预测的 LLM 驱动框架</h1> 
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
  <a href="https://github.com/Xiaoyu-Tao/TokenCast/stargazers">
    <img src="https://img.shields.io/github/stars/Xiaoyu-Tao/TokenCast?style=social" alt="Stars">
  </a>
  <a href="https://github.com/Xiaoyu-Tao/TokenCast/pulls">
    <img src="https://img.shields.io/badge/PRs-Welcome-green" alt="PRs Welcome">
  </a>

</div>

---

TokenCast 是一个创新框架，它利用 **大型语言模型（LLMs）** 进行 **上下文感知的时间序列预测**，方法是将连续时间序列转换为离散的符号 token。它实现了对时间与文本两种模态的统一生成建模。

> 📝 “From Values to Tokens: An LLM-Driven Framework for Context-aware Time Series Forecasting via Symbolic Discretization”
> **审稿中** | [📄 论文](https://arxiv.org/abs/2508.09191)

---

## 🔍 概述

传统预测模型难以有效整合临床记录、政策文档、日志等异构上下文数据。TokenCast 引入了一种新的范式：

* 通过动态向量量化将时间序列转换为**离散时间 token**。
* 使用冻结的预训练 LLM 将时间与文本 token **共同嵌入到共享语义空间**。
* 通过自回归语言建模实现 **基于 Prompt 的生成式预测**。

<p align="center">
  <img src="assets/main.png" width="700">
</p>

---

## ✨ 核心特性

* ✅ **离散化时间建模**：可学习、可逆的符号化时间序列 tokenizer
* 🔗 **跨模态对齐**：时间与文本 token 共享统一词表空间
* 📈 **Prompt 驱动生成**：通过 token 级指令生成进行预测
* 📊 **多领域评测**：涵盖经济、医疗、网络、股票与环境等领域
* 🌡️ **不确定性量化**：支持基于温度控制生成的预测区间

<!-- ---

## 📁 Project Structure
```

TokenCast/
 ├── tokenizer/               # Time series VQ-VAE tokenizer
 ├── models/                  # LLM backbone and embedding alignment
 ├── prompts/                 # Prompt templates for generation
 ├── datasets/                # Preprocessed benchmark datasets
 ├── evaluation/              # Evaluation scripts and metrics
 ├── scripts/                 # Training and inference scripts
 ├── configs/                 # YAML config files
 └── README.md

```
--- -->

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Xiaoyu-Tao/TokenCast.git
cd TokenCast
```

### 2. 环境配置

```bash
conda create -n tokencast python=3.10
conda activate tokencast
pip install -r requirements.txt
```

### 3. 准备数据

TokenCast 支持多个公开数据集：

* **经济（FRED-MD）**
* **医疗（Covid-19 mobility）**
* **网络（Wikipedia pageviews）**
* **股票：NY & NA（NYSE/NASDAQ）**
* **自然环境（传感器数据）**

首先，我们实验中使用的训练与评估数据可在 [Google Drive](https://drive.google.com/file/d/1HOCE20FQgLl0xCv_dOmLcTbN1RCZWwqd/view?usp=drive_link) 下载。
然后，创建一个名为 `datasets` 的目录并将数据下载到其中。

```bash
mkdir datasets
```

### 4. 训练时间序列 Tokenizer

```bash
sh Tokenizer/scripts/Czelan.sh 
```

### 5. 与 LLM 对齐嵌入

```bash
sh scripts/pretrain/Czelan.sh  
```

### 6. 微调预测模型

```bash
sh scripts/finetune/Czelan.sh 
```

---

## 📊 基准测试结果

**完整结果：**
![table1](assets/1-main-results.png)

**消融实验结果：**
![table2](assets/2-ablation-results.png)

---

## 📚 引用格式

如果你觉得这个项目对你有帮助，请引用我们的论文：

```bibtex
@inproceedings{tao2026tokencast,
  title={From Values to Tokens: An LLM-Driven Framework for Context-aware Time Series Forecasting via Symbolic Discretization},
  author={Tao, Xiaoyu and Zhang, Shilong and Cheng, Mingyue and Wang, Daoyu and Pan, Tingyue and Pan, Bokai and Zhang, Changqing and Wang, Shijin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

---

## 🤝 致谢

本项目由以下机构研究人员开发：

* 🧠 中国科学技术大学（USTC）
* 🧮 天津大学
* 🗣️ 科大讯飞研究院

---

## 📬 联系我们

如有问题或合作意向，请联系：

* 🧑‍🏫 Mingyue Cheng（[mycheng@ustc.edu.cn](mailto:mycheng@ustc.edu.cn)）
* 🤖 Xiaoyu Tao（[txytiny@mail.ustc.edu.cn](mailto:txytiny@mail.ustc.edu.cn)）

---

## 📌 许可证

本项目基于 MIT 协议开源。

---

如需【中英对照版】或【更加技术化的翻译】我也可以提供。
