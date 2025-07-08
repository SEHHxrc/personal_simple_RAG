from Base.BaseLLMInterface import BaseLLM
from Base.ChatHistory import ChatHistory

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer


class LocalTranslateModelManager(BaseLLM):
    def __init__(self, model_path: str=None):
        super().__init__(model_path)
        self.load()

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.local_model_path, trust_remote_code=True)

    def answer(self, source_text: str, translate_history: ChatHistory = None, **gen_kwargs):
        if translate_history is not None:
            full_text = translate_history.to_prompt() + source_text
        else:
            full_text = source_text
        inputs = self.tokenizer(full_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, **gen_kwargs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if translate_history is not None:
            translate_history.append(source_text, response)
        return response, translate_history
