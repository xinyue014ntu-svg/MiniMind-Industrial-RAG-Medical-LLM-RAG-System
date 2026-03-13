import os
import argparse
import warnings
import torch

# 1. 环境与网络配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings('ignore')

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from eval_llm import init_model

# ==========================================
# 2. 极速挂载企业级 RAG 知识库 (ChromaDB)
# ==========================================
def setup_rag():
    print("🧠 [1/2] 正在极速挂载本地持久化医疗知识库 (ChromaDB)...")
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    
    # 直接读取刚才生成的文件夹，实现秒级启动！
    vector_db = Chroma(
        persist_directory="./chroma_medical_db", 
        embedding_function=embeddings
    )
    print(f"✅ 知识库挂载成功！当前库内包含 {vector_db._collection.count()} 条权威医疗文献。")
    return vector_db

# ==========================================
# 3. 主程序：全自动问答循环
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from', type=str, default='model') 
    parser.add_argument('--weight', type=str, default='full_sft')
    parser.add_argument('--lora_weight', type=str, default='lora_medical')
    parser.add_argument('--save_dir', type=str, default='out')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--use_moe', type=int, default=0)
    parser.add_argument('--inference_rope_scaling', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args, _ = parser.parse_known_args()

    # 挂载新大脑
    vector_db = setup_rag()

    print("🚀 [2/2] 正在唤醒 MiniMind 医疗大模型...")
    model, tokenizer = init_model(args)
    model.eval()

    print("\n" + "="*50)
    print("🏥 【赛博华佗 RAG 诊所 (企业全量版) 已开启】")
    print("输入你的症状，输入 'quit' 退出系统。")
    print("="*50)

    while True:
        user_question = input("\n🤒 [患者提问]: ")
        if user_question.strip().lower() in ['quit', 'exit', 'q']:
            print("👋 诊所已关闭，祝您健康！")
            break
        if not user_question.strip():
            continue

        # 步骤A：从 5000 条权威文献中极速检索
        docs_and_scores = vector_db.similarity_search(user_question, k=1)
        retrieved_context = docs_and_scores[0].page_content
        print(f"   (💡 后台精准检索到相关文献: {retrieved_context[:40]}...)")

        # 步骤B：大白话 Prompt，配合小模型智商
        final_prompt = f"根据资料：“{retrieved_context}” 请回答患者的问题：{user_question} 医生的建议是："

        # 步骤C：模型推理生成回答（已解除复读机惩罚封印）
        print("👨‍⚕️ [赛博华佗]: ", end="", flush=True)
        inputs = tokenizer(final_prompt, return_tensors='pt').to(args.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(response)

if __name__ == "__main__":
    main()
