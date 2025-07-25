# metinlerde bulunan fazla boşluğu kaldırma
# %%
text ="   Bu   metin   örneği   fazla   boşluk   içeriyor.  "

def remove_extra_whitespace(text):
    """
    Remove extra whitespace from the text.
    """
    return ' '.join(text.split())   

remove_extra_whitespace(text)
# %%
text = "Bu MEtin Büyük küçük KÜÇÜK"
# buyuk harfleri küçük harfe çevirme
def convert_to_lowercase(text):
    """
    Convert all characters in the text to lowercase.
    """
    return text.lower() 
convert_to_lowercase(text)
# %%
# noktalama işaretlerini kaldırma
text = "Bu metin, noktalama işaretleri içeriyor! Evet, gerçekten de; içeriyor."
import string
def remove_punctuation(text):
    """
    Remove punctuation from the text.
    """
    return text.translate(str.maketrans('', '', string.punctuation))   

remove_punctuation(text) 

# %%
# özel karakterleri kaldırma
import re
text = "Bu metin, özel karakterler içeriyor! @#&*()du##%"
def remove_special_characters(text):
    """
    Remove special characters from the text.
    """
    return ''.join(e for e in text if e.isalnum() or e.isspace())
cleanText = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
print(cleanText)  
print("------------------------")
remove_special_characters(text)
# %%
# yazım hatalarını düzeltme
from textblob import TextBlob
text = "this is a smaple text with speling erors."


def correct_spelling(text):
    """
    Correct spelling errors in the text.
    """
    return str(TextBlob(text).correct())

print(TextBlob(text).correct())
print("------------------------")
print(correct_spelling(text))


# %%
# html etiketlerini kaldırma
from bs4 import BeautifulSoup
html_text = "<p>Bu metin <strong>HTML</strong> etiketleri içeriyordu.</p>"
def remove_html_tags(text):
    """
    Remove HTML tags from the text.
    """
    soup = BeautifulSoup(text, "html.parser")# with bs parse and get text
    return soup.get_text()
print(remove_html_tags(html_text))

# %%
