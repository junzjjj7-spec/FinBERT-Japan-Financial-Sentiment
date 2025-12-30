# 📈 Enhanced FinBERT for Financial Sentiment Analysis
> 基于混合数据策略与架构改良的垂直领域金融情感分析模型

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)

## 🔗 资源直达 (Resources)

本项目的所有核心资产均已开源至 Hugging Face Hub，可直接下载或在线试用。

| 资产类型 | 名称 | 描述 | 链接 |
| :--- | :--- | :--- | :--- |
| **🏆 最佳模型** | `finbert-japan-hybrid-v5-golden-best` | 基于混合策略训练的最终 SOTA 模型 (F1: 0.846) | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/dgawghbuidw/finbert-japan-hybrid-v5-golden-best) |
| **🧠 魔改架构** | `finbert-japan-advanced-v2-fix` | 引入 Focal Loss 与层融合的架构改良版 | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/dgawghbuidw/finbert-japan-advanced-v2-fix) |
| **📚 黄金数据集** | `finbert-japan-hybrid-v5-golden` | 包含 20k 合成数据 + 5k 真实数据的终极清洗版本 | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-green)](https://huggingface.co/datasets/dgawghbuidw/finbert-japan-hybrid-v5-golden) |

> *注：点击上方徽章即可跳转至 Hugging Face 仓库查看模型权重与数据详情。*

## 📖 项目背景 (Background)
通用的 FinBERT 模型在特定市场（如日本科技股）和复杂语境下表现不佳，且真实标注数据极其匮乏。本项目通过构建 **"合成+真实"混合数据管线**，并对 BERT 架构进行 **深层特征融合与 Loss 优化**，显著提升了模型在真实金融新闻上的泛化能力。

## 🚀 核心创新 (Key Features)

### 1. 数据工程 (Data Engineering 2.0)
- **混合数据策略 (Hybrid Strategy)**: 结合 20k DeepSeek 合成数据 + 5k Yahoo Finance 真实流数据。
- **AI 辅助清洗**: 利用 LLM (CoT Prompt) 进行数据去噪与二次打标，准确率 >95%。
- **动态过采样**: 解决真实数据稀缺导致的 Domain Shift 问题。

### 2. 模型魔改 (Architecture Mods)
- **Mod 1: Enhanced Head**: 替换原生 Linear 层，引入 `BatchNorm` + `GELU` + `Dropout` 增强非线性表达。
- **Mod 2: Layer Fusion**: 融合 BERT 最后三层 (`-1`, `-2`, `-3`) 的 `[CLS]` 向量，捕获从浅层语义到深层逻辑的完整特征。
- **Mod 3: Focal Loss**: 引入 `Focal Loss` 替代交叉熵，解决金融新闻中严重的类别不平衡（利空样本稀缺）问题。

## 📊 实验结果 (Results)

| 模型版本 | 训练数据策略 | 架构优化 | 真实测试集 F1 (Test Set) |
| :--- | :--- | :--- | :--- |
| Baseline (ProsusAI) | 原始数据 | 原生架构 | 0.80 |
| V1 | 仅合成数据 | 原生架构 | 0.82 (虚高，泛化差) |
| **Final Model** | **混合数据 (Hybrid)** | **Fusion + Focal Loss** | **0.86+ (SOTA)** |

> *注：最终模型在包含 500 条纯真实新闻的独立测试集上进行了评估。*

## 🛠️ 项目结构 (Structure)

```text
├── data_pipeline/
│   ├── scrape_news.py      # yfinance 实时爬虫
│   └── data_labeling.py    # DeepSeek API 自动打标脚本
├── model/
│   └── modeling_custom.py  # 自定义 CustomFinBERT 类与 FocalLossTrainer
├── train2.py                # 训练主程序
├── train_modeling_custom.py   # modeling_custom.py训练主程序
└── requirements.txt
##全deepseek生成数据集--数据增强+清洗，错误率95%
<img width="681" height="423" alt="finbert-japan-v7-lr2e5" src="https://github.com/user-attachments/assets/e8368c45-ffe4-4d4b-a809-fd99b6d72301" />
##测试集 ：500条 (真实)验证集 ：500条 (真实）训练集：所有合成数据 + 剩余真实数据 (约 23,000）
<img width="678" height="130" alt="finbert-japan-hybrid-v5-golden" src="https://github.com/user-attachments/assets/adf99d0f-8227-4764-937a-b9aff37c4d81" />
##23k数据集modeling_custom修改过后
<img width="1353" height="562" alt="屏幕截图 2025-12-30 152341" src="https://github.com/user-attachments/assets/65495c13-4240-4b37-abc9-fd988788bea8" />


