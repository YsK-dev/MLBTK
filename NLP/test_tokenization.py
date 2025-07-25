# %%
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
text = "This is a sample text. It contains several sentences, and some punctuation!"
def tokenize_words(text):
    """
    Tokenize the text into words.
    """
    return word_tokenize(text)

tokenize_words(text)

# sentence tokenization
def tokenize_sentences(text):
    """
    Tokenize the text into sentences.
    """
    return nltk.sent_tokenize(text)
tokenize_sentences(text)

# %%

# %%
