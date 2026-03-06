import os
import json
import warnings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

warnings.filterwarnings('ignore')

def main():
    print("📦 [1/4] 正在读取本地医疗问答数据集...")
    
    file_path = "train_zh_0.json"
    dataset = []
    
    # 极其稳妥的本地大文件读取法
    with open(file_path, "r", encoding="utf-8") as f:
        # 判断它是标准 JSON 还是 JSONL (按行存储)
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            dataset = json.load(f)
        else:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
                    
    # 取前 5000 条进行架构测试（等你跑通了，把这行注释掉，就能跑满 195 万条数据！）
    dataset = dataset[:5000]
    
    print(f"✅ 成功加载 {len(dataset)} 条权威医疗数据！")
    print("📝 [2/4] 正在解析标准 SFT 结构化数据...")
    
    docs = []
    for item in dataset:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        
        query = f"{instruction} {input_text}".strip()
        content = f"【患者症状/问题】: {query}\n【权威解答/治疗方案】: {output_text}"
        
        doc = Document(
            page_content=content,
            metadata={"source": "medical_finetune"}
        )
        docs.append(doc)

    print("🧠 [3/4] 正在加载 Embedding 向量化模型...")
    # 强制 Embedding 模型走镜像下载（这个库是可以被拦截的）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

    print("💾 [4/4] 正在进行高维向量化并持久化写入 ChromaDB (大约需要几分钟)...")
    persist_directory = "./chroma_medical_db"
    
    vectordb = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    print(f"\n🎉 大功告成！持久化医疗知识库已建立，存储于: {persist_directory} 目录下。")

if __name__ == "__main__":
    main()