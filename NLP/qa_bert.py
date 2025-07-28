# %%
from transformers import BertTokenizer, BertForQuestionAnswering
import torch    
import warnings
warnings.filterwarnings("ignore")
# %%

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)


def answer_question(question, context):
    """"Function to answer a question based on a given context using BERT model.
    Args:
        question (str): The question to be answered.
        context (str): The context in which the answer is to be found.
    Returns:
        str: The answer extracted from the context.
    aim: Extracts the answer from the context based on the question using BERT model.
    """
    inputs = tokenizer(question, context, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits 
    
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1  # +1 because the end index is inclusive
    
    answer_tokens = input_ids[0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens)
    
    return answer

# Example usage
context = "The capital of France is Paris. It is known for its art, fashion, and culture."
context1 = "Paris is the capital city of France, known for its art, fashion, and culture."
context2 = "The Eiffel Tower is one of the most famous landmarks in Paris, France."
context = context1 + " " + context2
question = "What is the capital of France?"
answer = answer_question(question, context2)
print(f"Question: {question}")
print(f"Answer: {answer}")

# %%


# %%
