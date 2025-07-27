# %%
import pandas as pd

data = pd.read_csv("/Users/ysk/Desktop/BTK/NLP/spam.csv", encoding='latin-1')

# %%
data = data[["v1", "v2"]]
data.columns = ["label", "text"]
# %%
print(data.isna().sum())
print(data.shape)

# %%
import nltk
import re
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

nltk.download("wordnet")
nltk.download("omw-1.4")


# %%
text =list(data.text)
lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(len(text)):
    review = re.sub('[^a-zA-Z]', ' ', text[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)
# %%
print(corpus[:5])

# %%
data['text2'] = corpus
# show original and processed text
print(data[['text', 'text2']].head())
# %%
from sklearn.model_selection import train_test_split
X = data['text2']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train_vectorized, y_train)   

X_test_cv= vectorizer.transform(X_test)

# predict
y_pred = classifier.predict(X_test_cv)

#show results
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))  
"""
Accuracy: 0.9739910313901345
Classification Report:
               precision    recall  f1-score   support

         ham       0.98      0.99      0.99       965
        spam       0.95      0.85      0.90       150

    accuracy                           0.97      1115
   macro avg       0.96      0.92      0.94      1115
weighted avg       0.97      0.97      0.97      1115

Confusion Matrix:
 [[958   7]
 [ 22 128]]
 """
# %%
