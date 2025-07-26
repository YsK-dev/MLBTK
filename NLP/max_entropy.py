"""
classification: max_entropy
sentiment ananlysis: max_entropy
"""
#%%
from nltk.classify import MaxentClassifier

# train data
train_data = [
    ({"Love": True, "Hate": False}, "positive"),
    ({"happy": True, "sad": False}, "positive"),
    ({"angry": True, "calm": False}, "negative"),
    ({"excited": True, "bored": False}, "positive"),
    ({"frustrated": True, "satisfied": False}, "negative"),
    ({"joyful": True, "depressed": False}, "positive"),
    ({"content": True, "discontent": False}, "positive"),
    ({"disappointed": True, "pleased": False}, "negative"),
    ({"optimistic": True, "pessimistic": False}, "positive"),
    ({"hopeful": True, "despairing": False}, "positive"),
    ({"grateful": True, "resentful": False}, "positive"),
    ({"confident": True, "doubtful": False}, "positive"),
    ({"relaxed": True, "stressed": False}, "positive"),
    ({"curious": True, "indifferent": False}, "positive"),]

classifier = MaxentClassifier.train(train_data, algorithm='GIS', max_iter = 10,trace=0)   

def extract_features(sentence):
    words = set(sentence.lower().split())
    # Use all words from training data as features
    feature_words = set()
    for features, _ in train_data:
        feature_words.update([w.lower() for w in features.keys()])
    return {word: (word in words) for word in feature_words}

test_sentence = "I Hate this movie"
features = extract_features(test_sentence)
predicted_sentiment = classifier.classify(features)
print(f"Predicted sentiment: {predicted_sentiment}")
# %%
