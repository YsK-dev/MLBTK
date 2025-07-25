#%% 
# bag of words

from sklearn.feature_extraction.text import CountVectorizer
import string

def create_dataset():
    """
    Create a sample dataset of documents.
    """
    return [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    ]

vectorizer = CountVectorizer()
documents = create_dataset()
X = vectorizer.fit_transform(documents)

# Display the feature names and the bag of words representation
print("Feature Names:", vectorizer.get_feature_names_out())
print("Bag of Words Representation:\n", X.toarray())
 

# Display the shape of the bag of words matrix
print("Shape of Bag of Words Matrix:", X.shape)
# Display the vocabulary
print("Vocabulary:", vectorizer.vocabulary_)
# Display the count of each word in the first document
print("Word Counts in First Document:", X[0].toarray()[0])
# Display the count of each word in the second document
print("Word Counts in Second Document:", X[1].toarray()[0]) 

# %%
# bow With IMDB Dataset
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

imdbData=pd.read_csv("/Users/ysk/Desktop/BTK/NLP/IMDB Dataset.csv")
# Display the first few rows of the dataset
print(imdbData.head())

# %%
documents = imdbData['review']
labels = imdbData['sentiment']

def clean_text(text):
    #Big letters to small
    text = text.lower()
    #Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    #Remove special characters
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    #remove extra spaces
    text = ' '.join(text.split())
    #remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    return text

# Clean the documents
cleaned_documents = [clean_text(doc) for doc in documents]  

# show cleaned documents
print("Cleaned Documents:")
for doc in cleaned_documents[:5]:  # Display first 5 cleaned documents
    print(doc)  

# %%

# Create a CountVectorizer instance
vectorizer = CountVectorizer()
# Fit and transform the cleaned documents
X = vectorizer.fit_transform(cleaned_documents)