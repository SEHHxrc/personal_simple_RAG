import argparse
from retriever import FaissRetriever
from embedder import LocalEmbedder
from Model.chat_local_model import LocalCausalModelManager
from Model.rag_local_model import RAGLLM
from Model.translate_local_model import LocalTranslateModelManager
from Model.ner_local_model import LocalNERModelManager
from Model.sentiment_local_model import LocalSentimentModelManager
from Base.ChatHistory import ChatHistory

from Middle.file_path_check import *


def main():
    parser = argparse.ArgumentParser(description="Local LLM CLI with optional RAG")
    parser.add_argument("--model", type=str, choices=["causal", "rag", "translation", "ner", "sentiment"], default="causal", help="Choose model type.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to local huggingface model.")
    parser.add_argument("--use-history", action="store_true", help="Enable multi-turn dialogue history.")
    parser.add_argument("--save-history-path", type=str, help="Optional: save chat history to file.")
    args = parser.parse_args()

    try:
        check_model_path(args.model_path)
    except ModelNotFoundError as e:
        print(f'{e}')
        exit(1)

    try:
        check_path(args.save_history_path, True, True)
    except FileNotFoundError:
        print("Not found history file, do not use history.")

    history = ChatHistory(save_path=args.save_history_path) if args.use_history else ChatHistory(max_turns=0)

    # 模型初始化
    if args.model == "rag":
        embedder = LocalEmbedder()
        retriever = FaissRetriever(embedder)
        retriever.load()
        llm = RAGLLM(args.model_path, retriever, history=history)
    elif args.model == "causal":
        llm = LocalCausalModelManager(args.model_path, history=history)
    elif args.model == "translation":
        llm = LocalTranslateModelManager(args.model_path)
    elif args.model == "ner":
        llm = LocalNERModelManager(args.model_path)
    elif args.model == "sentiment":
        llm = LocalSentimentModelManager(args.model_path)
    else:
        raise ValueError("Unsupported model type.")

    print(f"✅ 启动模型 [{args.model}]，输入 'exit' 退出对话。")
    while True:
        user_input = input("\nyou：")
        if user_input.lower() in ["exit", "quit"]:
            print("End")
            break
        try:
            response = llm.generate(user_input)
            print(f"\nassistant：{response}")
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
