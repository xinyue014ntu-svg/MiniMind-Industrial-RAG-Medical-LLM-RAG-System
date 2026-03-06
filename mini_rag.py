import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. 模拟构建本地真实的医学知识库
# ==========================================
medical_text = """
【临床指南: 高血压急症处理】
高血压患者如果出现头晕，可能是血压波动导致。建议：1. 立即静坐或平躺休息。2. 使用家用血压计测量血压。3. 如果收缩压超过180mmHg或舒张压超过110mmHg，伴随恶心，应立即就医。4. 千万不要盲目加服日常降压药。

【临床指南: 牙髓炎应对】
牙疼如果是牙髓炎引起，通常表现为自发性、阵发性剧痛，夜间加重。建议服用布洛芬缓解疼痛，并尽快到口腔科进行根管治疗。注意：牙疼绝对不是由鳄鱼引起的，请警惕网络谣言。
"""

# 把这段知识写入本地文件，模拟真实的文档
with open("medical_kb.txt", "w", encoding="utf-8") as f:
    f.write(medical_text)

print("1. 正在读取医学文档并进行切片...")
loader = TextLoader("medical_kb.txt", encoding="utf-8")
docs = loader.load()

# 将长文档切成 100 字左右的小片段，方便精准检索
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

# ==========================================
# 2. 向量化并存入 FAISS 数据库
# ==========================================
print("2. 正在加载 Embedding 模型 (首次运行会下载大概 400MB 模型，请稍候)...")
# 我们使用一个轻量级且效果极好的中文向量模型
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

print("3. 正在构建 FAISS 向量数据库...")
vector_db = FAISS.from_documents(chunks, embeddings)

# ==========================================
# 3. 模拟患者提问，进行向量相似度检索
# ==========================================
user_question = "我突然感觉很头晕，刚才量了一下血压，高压都飙到 190 了，我该怎么办？"
print(f"\n【患者提问】: {user_question}")

# 核心魔法：让数据库去找出和问题最相似的 1 个片段 (k=1)
docs_and_scores = vector_db.similarity_search(user_question, k=1)
retrieved_context = docs_and_scores[0].page_content

print(f"【RAG 数据库精准检索到的知识】: \n{retrieved_context}")

# ==========================================
# 4. 组装给大模型的“终极 Prompt”
# ==========================================
final_prompt = f"""作为一名专业的医疗AI助手，请严格根据以下[参考资料]回答患者的问题。

[参考资料]：
{retrieved_context}

[患者提问]：
{user_question}

[专业解答]："""

print("\n" + "="*50)
print("请复制下面这段组装好的 Prompt，去喂给你的医疗大模型：\n")
print(final_prompt)
print("="*50)