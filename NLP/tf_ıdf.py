# %%

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Köpek tatlı bir hayvan.",
    "Köpek ler motor cuları kovalar.",
    "motor ruhu taşır",
    "Güvenli motor sürün bir ömür boyu sürün ve köpekleri de sevin.",
]   

# tfidf vectorizer
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(documents)

# Display the feature names and the TF-IDF representation

print("TF-IDF Representation:\n", X.toarray())

df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())    

average_tfidf = df_tfidf.mean(axis=0)
print("\nAverage TF-IDF Scores (sorted):")
# Sort by score descending
for word, score in sorted(average_tfidf.items(), key=lambda x: x[1], reverse=True):
    print(f"{word}: {score:.4f}")
 
# %%
# with spam dataset tf-idf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
spam_data = pd.read_csv("/Users/ysk/Desktop/BTK/NLP/spam.csv", encoding='latin-1')
# Display the first few rows of the dataset
print(spam_data.head())
# %%
# clean the data
spam_data = spam_data[['v2', 'v1']].rename(columns={'v2': 'text', 'v1': 'label'});
spam_data['text'] = spam_data['text'].astype(str)  # Ensure text is string type
spam_data['label'] = spam_data['label'].astype(str)  # Ensure label is string type
# Remove any leading/trailing whitespace
spam_data['text'] = spam_data['text'].str.strip()
spam_data['label'] = spam_data['label'].str.strip()
# Remove any empty rows
spam_data = spam_data[spam_data['text'] != '']

# Remove any rows with NaN values
spam_data = spam_data.dropna(subset=['text', 'label'])

# Display the cleaned data
print(spam_data.head())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(spam_data['text'])

# Display the feature names and the TF-IDF representation
print("TF-IDF Representation:\n", X.toarray())

df_tfidf_spam = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
average_tfidf_spam = df_tfidf_spam.mean(axis=0)
print("\nAverage TF-IDF Scores (sorted):")
# Sort by score descending
for word, score in sorted(average_tfidf_spam.items(), key=lambda x: x[1], reverse=False):
    print(f"{word}: {score:.4f}")

#

# %%
