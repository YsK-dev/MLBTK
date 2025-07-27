# %%
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# %%
def generate_text(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# %%
# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt, max_length=100)
print("Generated text:", generated_text)
# %%
# llama2
try:
    from transformers import LlamaTokenizer, LlamaForCausalLM
except ImportError as e:
    print("LlamaTokenizer requires the 'sentencepiece' library. Install it with:")
    print("    pip install sentencepiece")
    raise e

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)    

# %%
