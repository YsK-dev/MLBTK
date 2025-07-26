"""
word2vec
fastext
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA 
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess

 
# %%
# create data
sentences = [
    "motorcu haksız",
    "yine bir motor kazası oldu otomobil sürücüsü hatalı",
    "motorcu haksız yere ceza yedi",
    "motorcu haksız yere ceza yedi ve kaza geçirdi",
    "otomobil sürücüsü çok dikkatliydi ve motorcuya çarpmadı",
    "otomobil çok hızlı motor çok yavaştı",
    "motorcu bir ömür boyu sürmek istiyor",
    "otomobil sürücüsü motorcuya çarpmadı ama motorcu yine de kaza geçirdi"
]
# Preprocess sentences
tok_sentences = [simple_preprocess(sentence) for sentence in sentences] 
# %%
Word2Vec_model = Word2Vec(sentences=tok_sentences, vector_size=50, window=5, min_count=1, workers=4)
print("Vocabulary:", Word2Vec_model.wv.index_to_key)
# %%
#fastext model
FastText_model = FastText(sentences=tok_sentences, vector_size=50, window=5, min_count=1, workers=4)
print("Vocabulary:", FastText_model.wv.index_to_key)
# %%
# Visualize Word2Vec model
def visualize_word2vec(model):
    words = list(model.wv.index_to_key)
    word_vectors = model.wv[words]
    
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(word_vectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
    
    for i, word in enumerate(words):
        plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1],))
    
    plt.title("Word2Vec Word Embeddings Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.show()
visualize_word2vec(Word2Vec_model)
# Visualize FastText model
def visualize_fasttext(model):
    words = list(model.wv.index_to_key)
    word_vectors = model.wv[words]
    
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(word_vectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
    
    for i, word in enumerate(words):
        plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1],))
    
    plt.title("FastText Word Embeddings Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.show()
visualize_fasttext(FastText_model)  
# %%
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import string
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
df =pd.read_csv("/Users/ysk/Desktop/BTK/NLP/IMDB Dataset.csv", encoding='latin-1')

documents =df['review']
# Clean the text data
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove special characters
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    # Remove extra spaces
    text = ' '.join(text.split())
    #remove english stp words
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    return text

    # Use only 200 samples from the data
sample_documents = documents[:200]

clean_texts = [clean_text(str(doc)) for doc in sample_documents]

    # Tokenize the cleaned texts
tok_sentences = [simple_preprocess(sentence) for sentence in clean_texts]


#
# %%
import matplotlib.pyplot as plt
# Train Word2Vec model
Word2Vec_model = Word2Vec(sentences=tok_sentences, vector_size=50, window=5, min_count=1, workers=4)
word_vectrors = Word2Vec_model.wv
words =list(word_vectrors.index_to_key)[:200]  # Limit to first 500 words for visualization

vectors =[word_vectrors[word] for word in words]

#clustering kmeans K=2
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(vectors) 
clusters = kmeans.labels_

#pca
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# Visualize clusters
plt.figure(figsize=(10, 8))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='viridis', alpha=0.5)

centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
for i, word in enumerate(words):  # Use the same 'words' list as for vectors
    plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=8)
plt.title("Word2Vec Clusters Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid()
plt.show()

# Train FastTex
# %%
