# %%
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# Download the VADER lexicon if not already installed
nltk.download('vader_lexicon'),
nltk.download('punkt'),
nltk.download('stopwords')
# %%
# Load the dataset
data = pd.read_csv('/Users/ysk/Desktop/BTK/NLP/IMDB Dataset.csv')  # Assuming the dataset is in CSV
#show the first few rows
print(data.head())

# %%
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(tokens)
# Preprocess the text in the dataset
data['review'] = data['review'].apply(preprocess_text)

# %%
analayzer = SentimentIntensityAnalyzer()
# Function to get sentiment score
def get_sentiment_score(text):
    score = analayzer.polarity_scores(text)
    
    return score['compound']  # Return the compound score  

get_sentiment_score('This is a great movie!')  # Example usage
df["sentiment_score"] = data['review'].apply(get_sentiment_score)


# %%
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(data['sentiment'], df['sentiment_score'] > 0.05))  # Assuming positive sentiment is > 0.05
print(confusion_matrix(data['sentiment'], df['sentiment_score'] > 0.05))  # Assuming positive sentiment is > 0.05   
# %%
