# from Model.translate_local_model import LocalTranslateModelManager
#
# # pass
# zh2en_translate_model = LocalTranslateModelManager('/home/sehh/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6')
# answer, _ = zh2en_translate_model.answer('我生而为王')
# print(answer)
#
# # pass
# en2zh_translate_model = LocalTranslateModelManager('/home/sehh/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-zh/snapshots/408d9bc410a388e1d9aef112a2daba955b945255')
# answer, _ = en2zh_translate_model.answer('I was born king')
# print(answer)
#
#
# from Model.ner_local_model import LocalNERModelManager
#
# # pass
# ner_model = LocalNERModelManager('/home/sehh/.cache/huggingface/hub/models--dbmdz--bert-large-cased-finetuned-conll03-english/snapshots/4c534963167c08d4b8ff1f88733cf2930f86add0')
# answer, _ = ner_model.answer('Barack Obama was born in Hawaii.')
# print(answer)
#
# # ner1_model = LocalNERModelManager('/home/sehh/.cache/huggingface/hub/models--uer--roberta-base-finetuned-cluener2020-chinese/snapshots/cddd8fc233e373855a8c0a7f4b7eb83acb686a2')
# # answer, _ = ner1_model.answer('我生而为王')
# # print(answer)
#
#
# from Model.summarization_local_model import LocalSummarizationModelManager
#
# # pass
# summary = LocalSummarizationModelManager('/home/sehh/.cache/huggingface/hub/models--facebook--bart-large-cnn/snapshots/37f520fa929c961707657b28798b30c003dd100b')
# answer, _ = summary.answer('The Transformers library developed by Hugging Face provides thousands of pre-trained models...It is widely used in the NLP community for tasks such as classification, summarization, translation, and more.', None, max_length=45, min_length=15, do_sample=False)
# print(answer)
#
# from Model.sentiment_local_model import LocalSentimentModelManager
#
# # pass
# analysis = LocalSentimentModelManager('/home/sehh/.cache/huggingface/hub/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13')
# answer, _ = analysis.answer('I was born king')
# print(answer)

from Model.chat_local_model import LocalCausalModelManager
from Base.ChatHistory import ChatHistory

model_path = '/home/sehh/.cache/huggingface/hub/models--Qwen--Qwen1.5-0.5B-Chat/snapshots/4d14e384a4b037942bb3f3016665157c8bcb70ea'
history = ChatHistory(5)
chat = LocalCausalModelManager(model_path, history)

inputs = ['你是谁', '你从哪里来', '你要到哪里去', '还记得我都问了什么吗']
for i in inputs:
    answer = chat.answer(i)
    print(answer)
