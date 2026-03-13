import os
import argparse
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 环境配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings('ignore')

def setup_rag():
    print("🧠 [1/2] 正在极速挂载本地持久化医疗知识库 (ChromaDB)...")
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    vector_db = Chroma(persist_directory="./chroma_medical_db", embedding_function=embeddings)
    print(f"✅ 知识库挂载成功！当前库内包含 {vector_db._collection.count()} 条权威医疗文献。")
    return vector_db

def main():
    # 挂载外挂大脑（检索部分代码完全不变，这就是解耦的作用。）
    vector_db = setup_rag()

    print("🚀 [2/2] 正在唤醒科室主任 (Qwen2.5-7B-Instruct) ... 这可能需要约 14GB 显存，请耐心等待。")
    # 替换为真实的工业级大模型路径（选择至少 24GB 显存的卡，比如 RTX 3090/4090）
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 工业级模型加载：使用 bfloat16 精度节省显存，并自动分配设备
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    ).eval()

    print("\n" + "="*50)
    print("🏥 【赛博华佗 RAG 诊所 (Qwen 7B 降维打击版) 已开启】")
    print("输入你的症状，输入 'quit' 退出系统。")
    print("="*50)

    while True:
        user_question = input("\n🤒 [患者提问]: ")
        if user_question.strip().lower() in ['quit', 'exit', 'q']:
            print("👋 诊所已关闭，祝您健康！")
            break
        if not user_question.strip():
            continue

        # 步骤A：依然从知识库中极速检索
        docs_and_scores = vector_db.similarity_search(user_question, k=1)
        retrieved_context = docs_and_scores[0].page_content if docs_and_scores else "未检索到相关资料"
        print(f"   (💡 后台精准检索到相关文献: {retrieved_context[:40]}...)")

        # 步骤B：使用标准的 System Prompt 规范约束大模型行为
        messages = [
            {"role": "system", "content": "你是一位专业且耐心的三甲医院主治医师。请务必基于提供的【参考资料】来回答患者的问题。如果参考资料中没有相关信息，请明确告知患者，切勿凭借自身基础知识胡编乱造（避免幻觉）。"},
            {"role": "user", "content": f"【参考资料】：\n{retrieved_context}\n\n【患者问题】：\n{user_question}\n\n请给出你的专业建议："}
        ]
        
        # 使用模型自带的 Chat Template 组装输入
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 步骤C：模型推理生成
        print("👨‍⚕️ [科室主任]: ", end="", flush=True)
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                temperature=0.3, # 医疗场景需要严谨，温度调低
                top_p=0.85,
                repetition_penalty=1.05
            )
            
        # 裁剪掉 Prompt 部分，只输出生成的回复
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)

if __name__ == "__main__":
    main()
