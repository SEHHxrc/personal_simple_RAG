from abc import ABC, abstractmethod
import torch
import os

from Base.ModelException import ModelNotFoundError


class BaseLLM(ABC):
    def __init__(self, model_path: str, device: str='cpu'):
        if not os.path.exists(model_path):
            raise ModelNotFoundError()
        self.local_model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None


    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def answer(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def user_format(user_format: str, words: str):
        if user_format == 'openai':
            return [{'role': 'user', 'content': words}]
        elif user_format == 'chatml':
            return f"<|user|>\n{words}\n<|assistant|>\n"
        elif user_format == 'plain':
            return f"role: {words}"
        elif user_format == 'chinese plain':
            return f"用户：{words}"
        else:
            raise ValueError('No Such template.')
