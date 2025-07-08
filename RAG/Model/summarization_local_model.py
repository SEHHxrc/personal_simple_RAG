from Base.BaseLLMInterface import BaseLLM
from Base.ChatHistory import ChatHistory

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer



class LocalSummarizationModelManager(BaseLLM):
    def __init__(self, model_path: str, device: str=None):
        super().__init__(model_path, device)
        self.load()

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.local_model_path, trust_remote_code=True)

    def answer(self, text: str, history: ChatHistory=None, history_format: str='openai', **gen_kwargs):
        if history is not None and not history.is_empty():
            prompt = history.to_prompt(history_format)
            full_text = self.tokenizer.apply_chat_template(
                prompt + self.user_format(history_format, text),
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            full_text = self.tokenizer.apply_chat_template(
                self.user_format(history_format, text),
                tokenize=False,
                add_generation_prompt=True
            )
        inputs = self.tokenizer(full_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, **gen_kwargs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if history is not None:
            history.append(text, response)
        return response, history