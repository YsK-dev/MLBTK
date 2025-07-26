# %%
# N-Gram Analysis
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

doc = [
    "motorcu haksız",
    "yine bir motor kazası oldu otomobil sürücüsü hatalı",
    "motorcu haksız yere ceza yedi",
    "motorcu haksız yere ceza yedi ve kaza geçirdi",
    "otomobil sürücüsü çok dikkatliydi ve motorcuya çarpmadı",
    "otomobil çok hızlı motor çok yavaştı",
    "motorcu bir ömür boyu sürmek istiyor",
    "otomobil sürücüsü motorcuya çarpmadı ama motorcu yine de kaza geçirdi"
]

vectorizer_ngram = CountVectorizer(ngram_range=(1, 1))

vectorizer_bigram = CountVectorizer(ngram_range=(2, 2))

vectorizer_trigram = CountVectorizer(ngram_range=(3, 3))

# unigram
X_ngram = vectorizer_ngram.fit_transform(doc)
print("Unigram Feature Names:", vectorizer_ngram.get_feature_names_out())
print("Unigram Representation:\n", X_ngram.toarray())   

# %%
# bigram
X_bigram = vectorizer_bigram.fit_transform(doc)
print("Bigram Feature Names:", vectorizer_bigram.get_feature_names_out())
print("Bigram Representation:\n", X_bigram.toarray())   
# %%
# trigram
X_trigram = vectorizer_trigram.fit_transform(doc)
print("Trigram Feature Names:", vectorizer_trigram.get_feature_names_out())
print("Trigram Representation:\n", X_trigram.toarray()) 
# %%
print("Shape of Unigram Matrix:", X_ngram.shape)
print("Shape of Bigram Matrix:", X_bigram.shape)
print("Shape of Trigram Matrix:", X_trigram.shape)
# %%
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from collections import Counter

corpus = [
    "motorcu haksız",
    "yine bir motor kazası oldu otomobil sürücüsü hatalı",
    "motorcu haksız yere ceza yedi",
    "motorcu haksız yere ceza yedi ve kaza geçirdi",
    "otomobil sürücüsü çok dikkatliydi ve motorcuya çarpmadı",
    "otomobil çok hızlı motor çok yavaştı",
    "motorcu bir ömür boyu sürmek istiyor",
    "otomobil sürücüsü motorcuya çarpmadı ama motorcu yine de kaza geçirdi"
]

# what we want is to guess the next word based on the previous words for that use n-grams

tok = [word_tokenize(doc.lower()) for doc in corpus]

# %%
# bigram
bigrams = [ngrams(doc, 2) for doc in tok]
bigram_freq = Counter([bigram for sublist in bigrams for bigram in sublist])
print("Bigram Frequencies:", bigram_freq)


# %%

#3-gram
trigrams = [ngrams(doc, 3) for doc in tok]
trigram_freq = Counter([trigram for sublist in trigrams for trigram in sublist])
print("Trigram Frequencies:", trigram_freq) 
# %%
# model testing 
# we will use the bigram model to predict the next word
def predict_next_word(bigram_freq, previous_word):
    candidates = {bigram[1]: freq for bigram, freq in bigram_freq.items() if bigram[0] == previous_word}
    if not candidates:
        return None
    return max(candidates, key=candidates.get)  
predict_next_word(bigram_freq, 'otomobil')  # Example usage
# show percentage of each word in the bigram
def bigram_percentage(bigram_freq):
    total_count = sum(bigram_freq.values())
    return {bigram: (freq / total_count) * 100 for bigram, freq in bigram_freq.items()}
bigram_percentages = bigram_percentage(bigram_freq)
print("Bigram Percentages:", bigram_percentages)

# %%
