from Base.BaseLLMInterface import BaseLLM
from Base.ChatHistory import ChatHistory

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


class LocalCausalModelManager(BaseLLM):

    def __init__(self, local_model_path: str, history: ChatHistory=None, device=None):
        super().__init__(local_model_path, device)
        self.history = history
        self.load()

    def load(self):
        self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True).to(self.device).eval()

    def answer(self, user_input, max_new_tokens=200, history_format='openai'):
        if self.history is not None and not self.history.is_empty():
            prompt = self.history.to_prompt(history_format)
            input_text = self.tokenizer.apply_chat_template(
                prompt+self.user_format(history_format, user_input),
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = self.tokenizer.apply_chat_template(
                self.user_format(history_format, user_input),
                tokenize=False,
                add_generation_prompt=True
            )
        print(input_text)
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        if self.history is not None:
            self.history.append(user_input, response)
        return response
