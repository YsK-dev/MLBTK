"""
part of speech tagging using a Hidden Markov Model (HMM) in Python.
"""
# %%
import nltk
from nltk.tag import hmm

# Sample training data: sentences with their corresponding part-of-speech tags
train_data = [
    (["the", "dog", "barked"], ["DT", "NN", "VBD"]),
    (["the", "cat", "meowed"], ["DT", "NN", "VBD"]),
    (["the", "bird", "sang"], ["DT", "NN", "VBD"]),
    (["the", "dog", "chased", "the", "cat"], ["DT", "NN", "VBD", "DT", "NN"]),
    (["the", "cat", "sat", "on", "the", "mat"], ["DT", "NN", "VBD", "IN", "DT", "NN"]), 
    (["the", "bird", "flew", "away"], ["DT", "NN", "VBD", "RP"]),
]
# Train the HMM 
trainer = hmm.HiddenMarkovModelTrainer()
hmm_model = trainer.train(train_data)
# %%
# Sample test sentence
test_sentence = ["the", "dog", "sat", "on", "the", "mat"]
# Predict the part-of-speech tags for the test sentence
predicted_tags = hmm_model.tag(test_sentence)
print("Predicted Tags:", predicted_tags)
# %%
# hmm with big data
import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000

nltk.download('conll2000')
# %

# Load the training data from the CoNLL 2000 corpus
def tree_to_sequences(chunked_sents):
    sequences = []
    for tree in chunked_sents:
        # Convert each tree to a list of (word, tag) tuples
        sequences.append([(word, tag) for word, tag in tree.pos()])
    return sequences

train_chunked = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
test_chunked = conll2000.chunked_sents('test.txt', chunk_types=['NP'])

train_data = tree_to_sequences(train_chunked)
test_data = tree_to_sequences(test_chunked)

print("Training data size:", len(train_data))
print("Test data size:", len(test_data))

# Train the HMM model
trainer = hmm.HiddenMarkovModelTrainer()
hmm_model = trainer.train(train_data)
# Train the 
# %%
