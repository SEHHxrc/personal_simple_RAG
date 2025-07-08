from chat_local_model import LocalCausalModelManager
from Base.ChatHistory import ChatHistory
from retriever import FaissRetriever


class RAGLLM(LocalCausalModelManager):
    def __init__(self, model_path: str, retriever: FaissRetriever, history: ChatHistory = None, device: str = None):
        super().__init__(model_path, history, device)
        self.retriever = retriever

    def generate(self, user_input: str, max_new_tokens: int = 128, top_k: int = 3, history_format: str='chatml', **gen_kwargs):
        retrieved = self.retriever.retrieve(user_input, top_k=top_k)
        context = "\n".join([f"参考资料：{doc['text']}" for doc in retrieved])
        if self.history is not None and not self.history.is_empty():
            full_prompt = context + "\n" + self.history.to_prompt(
                history_format) + self.user_format(history_format, user_input)
        else:
            full_prompt = context + "\n" + self.user_format(history_format, user_input)
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
        result = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        self.history.append(user_input, result)
        return result
