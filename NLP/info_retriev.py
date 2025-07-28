# %%
from transformers import BertTokenizer, BertForQuestionAnswering,BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_name = "bert-base-uncased" #small model for faster inference

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

documents = [

    "Machine learning is a subset of artificial intelligence.",
    "Deep learning is a subset of machine learning.",
    "Natural language processing is a field of artificial intelligence.",
    "Computer vision is a field of artificial intelligence.",
    "Reinforcement learning is a type of machine learning.",
    "Reinforcement learning is used in robotics.",
    "Supervised learning is a type of machine learning.",
    "Unsupervised learning is a type of machine learning.",
    "Semi-supervised learning is a type of machine learning.",
    "Transfer learning is a technique in machine learning.",
    "Generative adversarial networks are used in deep learning.",
    "Convolutional neural networks are used in computer vision.",
    "Recurrent neural networks are used in natural language processing.",
    "Long short-term memory networks are a type of recurrent neural network.",
    "Transformers are a type of neural network architecture.",
    "Attention mechanisms are used in transformers.",
    "BERT is a transformer-based model for natural language processing.",
    "GPT is a transformer-based model for natural language processing.",
    "XLNet is a transformer-based model for natural language processing.",
    "RoBERTa is a variant of BERT.",
    "DistilBERT is a smaller version of BERT.",
    "ALBERT is a variant of BERT with fewer parameters.",
    "T5 is a transformer-based model for text-to-text tasks.",
    "I love programming in Python.",
    " What Versatile is I dont Know"
]

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True) # Tokenize and encode the texts
    with torch.no_grad():# Disable gradient calculation for inference
        outputs = model.bert(**inputs)# Get the outputs from the model
    # Use the mean of the last hidden state as the embedding
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings 


# %%

# documents an for question
documents_embeddings = get_embeddings(documents)
question = "What is natural language processing?"
question_embedding = get_embeddings([question])[0]
cosine_similarity_scores = cosine_similarity([question_embedding], documents_embeddings)[0]

print("Question Embedding:", question_embedding)
print("Document Embeddings:", documents_embeddings)
print("Cosine Similarity Scores:", cosine_similarity_scores)
# Get the index of the most similar document
most_similar_index = np.argmax(cosine_similarity_scores)
print("Most Similar Document Index:", most_similar_index)
print("Most Similar Document:", documents[most_similar_index])
# %%
