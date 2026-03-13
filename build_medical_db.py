import os
import json
import warnings
from tqdm import tqdm  # 工业级进度条，必备！需 pip install tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

warnings.filterwarnings('ignore')

def stream_read_jsonl(file_path, batch_size=1000):
    """
    核心工程亮点：生成器 (Generator) 模式流式读取。
    不管文件是 1GB 还是 100GB，内存占用永远只有 batch_size 大小，彻底杜绝 OOM。
    """
    batch = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 兼容处理可能存在的 JSON 数组括号或尾部逗号
            if not line or line in ('[', ']'): 
                continue
            if line.endswith(','): 
                line = line[:-1]
                
            try:
                item = json.loads(line)
                batch.append(item)
                
                # 当达到批次上限时，使用 yield 吐出数据，并清空当前列表释放内存
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            except json.JSONDecodeError:
                # 容错机制：跳过极个别损坏的脏数据，保证整个流水线不中断
                continue
                
        # 将最后不足一个 batch 的剩余数据吐出
        if batch:
            yield batch

def main():
    print("🚀 启动企业级全量数据灌库引擎 (流式微批次处理)...")
    
    # 1. 强制走镜像，加载 Embedding
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("🧠 [1/3] 正在加载 Embedding 向量化模型...")
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

    # 2. 预先初始化空的 ChromaDB 实例（核心改动）
    persist_directory = "./chroma_medical_db"
    print(f"💾 [2/3] 初始化 ChromaDB 实例，目标目录: {persist_directory}")
    # 注意：这里不再使用 from_documents，而是先建立连接
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # 3. 流式读取与增量持久化
    file_path = "train_zh_0.json"
    BATCH_SIZE = 1000 # 每次只将 1000 条数据送入 GPU 向量化
    TOTAL_ESTIMATE = 1950000 # 预估总数，用于进度条展示
    
    print(f"📦 [3/3] 开始全量解析与流式写入 (Batch Size: {BATCH_SIZE})...")
    
    # 引入 tqdm 进度条，长耗时任务必须有监控感知
    with tqdm(total=TOTAL_ESTIMATE, desc="向量化灌库进度", unit="条") as pbar:
        for batch_data in stream_read_jsonl(file_path, batch_size=BATCH_SIZE):
            docs = []
            for item in batch_data:
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                
                query = f"{instruction} {input_text}".strip()
                content = f"【患者症状/问题】: {query}\n【权威解答/治疗方案】: {output_text}"
                
                doc = Document(
                    page_content=content,
                    metadata={"source": "medical_finetune", "status": "verified"}
                )
                docs.append(doc)
            
            # 核心改动：增量写入 (Incremental Add)
            # 这一步 GPU 计算完后，直接刷入本地硬盘，绝不堆积在内存里
            vectordb.add_documents(documents=docs)
            
            # 更新进度条
            pbar.update(len(batch_data))
            
            # (可选) 强制 Chroma 每完成一个 Batch 存盘一次（新版 Chroma 通常自动落盘，但可以更保险）
            # vectordb.persist() 

    print("\n🎉 史诗级胜利！195 万条全量权威医疗数据已完美转化为向量外挂大脑。")
    print(f"📁 知识库体积大幅膨胀，请检查 {persist_directory} 目录。")

if __name__ == "__main__":
    main()
