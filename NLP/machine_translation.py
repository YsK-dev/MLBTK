# %%
! pip install sentencepiece
# %%
from transformers import MarianMTModel, MarianTokenizer
import torch

# %%
model_name = "Helsinki-NLP/opus-mt-en-de"  
"Load the model and tokenizer"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Hello, how are you?"

translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True, truncation=True))

translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
print(f"Translated text: {translated_text}")
# %%
