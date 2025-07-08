from Base.BaseLLMInterface import BaseLLM
from Base.ChatHistory import ChatHistory

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F


class LocalSentimentModelManager(BaseLLM):
    def __init__(self, model_path: str, device: str=None):
        super().__init__(model_path, device)
        self.load()

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.local_model_path, trust_remote_code=True)

    def answer(self, words: str, history: ChatHistory=None, history_format: str='openai', **kwargs):
        if history is not None and not history.is_empty():
            prompt = history.to_prompt(history_format)
            full_words = self.tokenizer.apply_chat_template(
                prompt + self.user_format(history_format, words),
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            full_words = self.tokenizer.apply_chat_template(
                self.user_format(history_format, words),
                tokenize=False,
                add_generation_prompt=True
            )

        inputs = self.tokenizer(full_words, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1)[0].cpu().tolist()

        label_id = int(torch.argmax(torch.tensor(probs)))
        label_name = self.model.config.id2label[label_id]
        if history is not None:
            history.append(words, label_name)
        return {"label": label_name, "score": probs[label_id]}, history
