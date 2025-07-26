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
