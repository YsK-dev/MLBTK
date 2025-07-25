#%% 
# bag of words

from sklearn.feature_extraction.text import CountVectorizer

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
