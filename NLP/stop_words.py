# english stop words analysis
#%%
import nltk
from nltk.corpus import stopwords

# Download the stopwords corpus
nltk.download('stopwords', quiet=True)

# Get the list of English stop words
stop_words = set(stopwords.words('english'))
# Print the first 20 stop words
print("First 20 English stop words:")
print(list(stop_words)[:20])    

text = "This is a sample text that contains some the common stop words like 'the', 'is', and 'in'."

text_list = text.split()
# Filter out stop words from the text
filtered_text = [word for word in text_list if word.lower() not in stop_words]
print("\nOriginal text:")
print(text)
print("\nFiltered text (without stop words):")
print(filtered_text)
# %%
from nltk.corpus import stopwords
import nltk
# Download the stopwords corpus for Turkish
nltk.download('stopwords', quiet=True)
# turkçe stop words analysis
stop_words_tr = set(stopwords.words('turkish'))
# Print the first 20 Turkish stop words
print("\nFirst 20 Turkish stop words:")
print(list(stop_words_tr)[:20])
text_tr = "Bu bir örnek metindir. İçinde 'bu', 'bir', ve 'için' gibi yaygın durak kelimeler bulunur."
text_list_tr = text_tr.split()
# Filter out stop words from the Turkish text
filtered_text_tr = [word for word in text_list_tr if word.lower() not in stop_words_tr]
print("\nOrijinal metin:")
print(text_tr)
print("\nDurak kelimeler olmadan filtrelenmiş metin:")
print(filtered_text_tr)
# %%
#p without library stop words analysis

text = "Bak arkadaş bu bir örnek metin. İçinde stop word budur dediğim kelimeler var by YsK"

def remove_stop_words(text):
    """
    Remove common stop words from the text.
    """
    stop_words = ["bu", "bir", "içinde", "var", "dediğim", "kelimeler", "by", "YsK"]
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

def show_stop_words(text):
    """
    Show the stop words in the text.
    """
    stop_words = ["bu", "bir", "içinde", "var", "dediğim", "kelimeler", "by", "YsK"]
    return [word for word in text.split() if word.lower() in stop_words]    
# Remove stop words from the text
filtered_text = remove_stop_words(text)
print("\nOriginal text:")
print(text)
print("\nFiltered text (without stop words):")
print(filtered_text)
# Show stop words in the text
stop_words_in_text = show_stop_words(text)
print("\nStop words in the text:")
print(stop_words_in_text)
# %%


