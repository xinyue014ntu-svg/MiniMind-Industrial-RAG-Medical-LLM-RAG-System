# 🏥 MiniMind-Medical-RAG: 极限算力下的医疗大模型微调与检索增强实践

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![LangChain](https://img.shields.io/badge/LangChain-Integration-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-green)

## 📖 项目简介

本项目旨在探索极端算力限制下的大语言模型（LLM）垂直落地能力。通过将极轻量级语言模型（MiniMind，25.96M）架构与 **企业级 RAG（检索增强生成）系统** 相结合，完整跑通了从“底层模型微调”到“海量知识库挂载”的全生命周期链路。

项目不仅攻克了垂直领域微调中的“灾难性遗忘”难题，还通过硬核解析 GB 级真实医疗百科数据与持久化向量数据库，成功构建了一个具备高维语义检索与定制化问答能力的“赛博医疗助手”。

## ✨ 核心工程亮点 (Engineering Highlights)

### 1. 垂直领域全链路微调 (Domain-Specific Fine-Tuning)
* **全链路跑通**：独立完成了环境配置、SFT 全量指令微调及特定医疗领域的 LoRA 权重训练。
* **攻克“灾难性遗忘”**：在初始数据注入时，模型出现了严重的语言能力崩坏（如输出乱码逻辑）。通过深入分析基础语言结构破坏现象，动态调整超参数（显著降低 Epochs 并优化 Learning Rate），成功在“注入新医学知识”与“保留原有语言能力”之间找到极佳的平衡点。

### 2. 突破网络瓶颈的 GB 级数据清洗 (Industrial Data Processing)
* **海量语料解析**：弃用常规且容易受限于网络墙的 `datasets` API 内存加载方式，通过底层命令直拉并解析 **1.2GB** 的 `shibing624/medical` 高质量医患问答 JSON 文件。
* **结构化提纯**：编写流式读取脚本，将复杂的嵌套 JSON 结构清洗为 `【患者症状/问题】+【权威解答/治疗方案】` 的标准检索文档格式，展现了处理工业级海量语料的扎实基本功。

### 3. 企业级向量数据库落地 (Enterprise RAG Architecture)
* **持久化存储**：摒弃了初学者常用的纯内存检索（如简单的 FAISS 内存实例），引入了支持本地持久化的高性能向量数据库 `ChromaDB`。
* **极速挂载与检索**：结合 `text2vec-base-chinese` 词向量模型，将数万条真实医疗文献高维向量化并固化至本地硬盘，实现了 RAG 外挂知识库的 **秒级极速挂载** 与高并发近似最近邻（ANN）查询。

### 4. 模型边界与注意力极限测试 (Boundary Testing & Decoupling)
* **高价值踩坑经验**：在系统级测试阶段，精准捕获并记录了 25M 极小参数模型在面对超长真实医疗文献时，由于“注意力窗口（Attention Window）超载”引发的输出乱码与彻底失语现象。
* **架构解耦思维**：验证了当前 RAG 检索链路（Retriever）的健壮性。外围 LangChain 逻辑无需改动，仅需将底座 Generator 模型无缝平替为 Llama-3 (8B) 或 Qwen-2.5 (7B) 等工业级基座，即可瞬间消除幻觉，具备极强的生产环境迁移与落地能力。

## 🛠️ 技术栈 (Tech Stack)

* **算法架构**: Transformer (Causal LM), LoRA, RAG (Retrieval-Augmented Generation)
* **核心框架**: PyTorch, HuggingFace Transformers, LangChain, ChromaDB
* **工业基座**: MiniMind (25M), Qwen2.5 (7B), text2vec-base-chinese
* **开发环境**: AutoDL (Linux), JupyterLab

## 📂 核心文件说明

* `eval_llm.py`：基础对话脚本。支持单模型推理与挂载 LoRA 权重的推理测试。
* `build_medical_db.py`：后台离线数据引擎。负责读取 GB 级 JSON 语料，进行长文本切片与向量化，并构建持久化 ChromaDB 目录。
* `chat_rag.py`：**在线 RAG 问答引擎**。采用解耦架构，无缝串联 Chroma 向量检索与 LLM 生成（支持 Qwen 7B ）。

## 🚀 快速启动 (Quick Start)

### 1. 构建外挂硬盘大脑
python build_medical_db.py

*(注意：请确保已准备好 `train_zh_0.json` 巨型语料文件，并在同级目录下运行)*



### 2. 启动赛博华佗全量诊所
python chat_rag.py --weight full_sft --lora_weight lora_medical

*(⚠️ 开发者提示：受限于 25M 极小参数架构，模型在处理超长真实 RAG 文献时可能出现解码乱码现象，该分支核心代码可无缝平替至 7B 以上级别工业模型。)*
