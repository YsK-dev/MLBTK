# %%
from transformers import AutoTokenizer, AutoModel
import torch

#model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
# %%
text = "Merhaba dünya! Hello world! merhaba optimus prime sana da merhaba megatron"

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad(): # Disable gradient calculation
    outputs = model(**inputs)
# Get the last hidden state 
last_hidden_state = outputs.last_hidden_state
# first token
first_token_embedding = last_hidden_state[0][0]
print(f"First token embedding: {first_token_embedding}")


# %%
