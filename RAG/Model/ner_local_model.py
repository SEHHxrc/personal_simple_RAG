from Base.BaseLLMInterface import BaseLLM
from Base.ChatHistory import ChatHistory

from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import torch


class LocalNERModelManager(BaseLLM):
    def __init__(self, model_path: str=None, device: str=None):
        super().__init__(model_path, device)
        self.load()

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.local_model_path, trust_remote_code=True)

    def answer(self, text: str, ner_history: ChatHistory=None, filter=False):
        if ner_history is not None:
            full_text = ner_history.to_prompt() + text
        else:
            full_text = text
        inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        outputs= self.model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities = [(token, self.model.config.id2label[prediction]) for token, prediction in zip(tokens, predictions)]
        if ner_history is not None:
            ner_history.append(text, entities)
        if filter:
            entities = self.__filter(entities)
        return entities, ner_history

    @staticmethod
    def __filter(entities):
        label_entities = {}
        for token, label in entities:
            if label == '0':
                continue
            elif label in label_entities.keys():
                label_entities[label].append(token)
            else:
                label_entities[label] = [token]
        return label_entities
