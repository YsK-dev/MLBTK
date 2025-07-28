# %%
from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization")

text = """Machine learning is a subset of artificial intelligence that focuses on the development of algorithms that allow computers to learn from and make predictions based on data. It involves the use of statistical techniques to enable machines to improve their performance on a specific task over time without being explicitly programmed. Machine learning is widely used in various applications, including natural language processing, computer vision, and recommendation systems. The field has seen significant advancements in recent years, driven by the availability of large datasets and increased computational power. Key techniques in machine learning include supervised learning, unsupervised learning, and reinforcement learning. Supervised learning involves training a model on labeled data, while unsupervised learning deals with unlabeled data to find patterns or group  similar data points. Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize a reward signal. The success of machine learning has led to its integration into various industries, transforming how businesses operate and making significant impacts on society."""

summ=summarizer(text, max_length=130, min_length=30, do_sample=False)
# %%
print("Summary:", summ[0]['summary_text'])
# %%
