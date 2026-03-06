# 🏥 Industrial-Medical-RAG: 基于海量数据的医疗大模型检索增强系统

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Integration-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-green)

## 📖 项目简介
本项目旨在探索算力极限下的大语言模型（LLM）垂直落地能力。通过将轻量级语言模型（MiniMind，25M）与 **企业级 RAG（检索增强生成）架构** 相结合，成功实现了基于海量真实医疗百科文献的精准问答。项目完整覆盖了从大文件硬解析、向量持久化到自动化工作流编排的全链路工程实践。

## ✨ 核心架构与工程亮点 (Engineering Highlights)

### 1. 突破网络瓶颈的 GB 级数据清洗
* 弃用常规的 `datasets` API 内存加载方式，通过底层命令直拉 **1.2GB** 的 `shibing624/medical` 高质量医患问答 JSON 文件。
* 编写流式读取脚本，将复杂的嵌套 JSON 结构清洗为 `【患者症状】+【权威解答】` 的标准检索文档格式，展现了处理工业级海量语料的扎实基本功。

### 2. 企业级向量数据库 (ChromaDB) 落地
* 摒弃了初学者常用的纯内存检索（如简单的 FAISS 内存实例），引入了支持本地持久化的 `ChromaDB`。
* 结合 `text2vec-base-chinese` 词向量模型，将数万条医疗文献高维向量化并固化至本地硬盘，实现了 RAG 知识库的 **秒级极速挂载** 与高并发近似最近邻（ANN）查询。

### 3. 模型边界与注意力极限测试 (Boundary Testing)
* **高价值踩坑经验**：在测试阶段，精准捕获并记录了极小参数模型在面对超长真实医疗文献时，由于“注意力窗口（Attention Window）超载”引发的输出乱码与彻底失语现象。
* **架构解耦**：验证了当前 RAG 检索链路（Retriever）的健壮性。外围 LangChain 逻辑无需改动，仅需将底座模型切换为 Llama-3 或 Qwen-2.5 等 7B 级别工业模型，即可瞬间消除幻觉，具备极强的生产环境平替能力。

## 🛠️ 核心文件说明
* `build_medical_db.py`：后台离线数据引擎。负责读取 GB 级 JSON 语料，进行长文本切片与向量化，并构建持久化 ChromaDB 目录。
* `chat_rag.py`：在线 RAG 问答引擎。无缝串联 Chroma 向量检索与 LLM 生成，内置大白话 Prompt 组装与复读机惩罚（Repetition Penalty）动态调参。

## 🚀 快速启动 (Quick Start)
1. 构建外挂硬盘大脑
*(注意：请确保已准备好 `train_zh_0.json` 语料文件)*
```bash
python build_medical_db.py

2. 启动赛博华佗全量诊所
```bash
python chat_rag.py --weight full_sft --lora_weight lora_medical
