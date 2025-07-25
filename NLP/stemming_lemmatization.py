
#%%
import nltk

nltk.download('wordnet', quiet=True)

from nltk.stem import PorterStemmer

# create a PorterStemmer object
stemmer = PorterStemmer()

words = ["running", "ran", "easily", "fairly","go","went", "happily", "happiness"]

def stem_words(words):
    """
    Stem the list of words using PorterStemmer.
    """
    return [stemmer.stem(wo) for wo in words]
# finding stems of the words using PorterStemmer
print("Original words:", words)
print("------------------------")
print("Stems of the words:")
stemmed_words = stem_words(words)
print("Stemmed words:", stemmed_words)  


# %%
# Lemmatisation using WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

words = ["running", "runner","ran", "easily","go","went", "fairly", "happily", "happiness"]
def lemmatize_words(words):
    """
    Lemmatize the list of words using WordNetLemmatizer.
    """
    return [lemmatizer.lemmatize(wo ,pos="v") for wo in words]

# finding lemmas of the words using WordNetLemmatizer
print("Original words:", words)
print("------------------------")
print("Lemmas of the words:")
lemmatized_words = lemmatize_words(words)
print("Lemmatized words:", lemmatized_words)
# %%
