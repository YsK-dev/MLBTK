


# %%
import nltk
from nltk.wsd import lesk

# Download the WordNet data if not already ;installed
nltk.download('wordnet')
nltk.download('omw-1.4')
    
nltk.download('punkt')

s1 = "The bank can guarantee deposits will eventually cover future tuition costs because it invests in adjustable-rate mortgage securities."
w1 = "bank"
s2 = "You can bank on the fact that the sun will rise tomorrow."
w2 = "bank"
# Using lesk to find the meaning of the word in context
context1 = nltk.word_tokenize(s1)
context2 = nltk.word_tokenize(s2)
meaning1 = lesk(context1, w1)
meaning2 = lesk(context2, w2)
print(f"Meaning of '{w1}' in context 1: {meaning1.definition() if meaning1 else 'Not found'}")
print(f"Meaning of '{w2}' in context 2: {meaning2.definition() if meaning2 else 'Not found'}")
"""
Meaning of 'bank' in context 1: a supply or stock held in reserve for future use (especially in emergencies)
Meaning of 'bank' in context 2: a financial institution that accepts deposits and channels the money into lending activities
"""


# %%

import nltk
nltk.download('averaged_perceptron_tagger')
from pywsd.lesk import adapted_lesk, simple_lesk, cosine_lesk;

sentence = "The bank can guarantee deposits will eventually cover future tuition costs because it invests in adjustable-rate mortgage securities."
word = "bank"
# Using adapted_lesk to find the meaning of the word in context
context = nltk.word_tokenize(sentence)
meaning = adapted_lesk(context, word)
print(f"Adapted Lesk meaning of '{word}': {meaning.definition() if meaning else 'Not found'}")


# %%
