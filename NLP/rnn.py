 # comments sentiment analysis
# %%
! pip install tensorflow
# %%
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN,Embedding, LSTM, Dense, Dropout  

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %%
# Truncate text list to match label length
data = {
    "text": [
        "Yemekler harikaydı, servis çok hızlıydı.",
        "Garson çok ilgisizdi, bir daha gelmem.",
        "Tatlı mükemmeldi ama ana yemek soğuktu.",
        "Mekan çok gürültülüydü, keyif alamadım.",
        "Servis hızlıydı, fiyatlar da makuldü.",
        "Porsiyonlar küçüktü, aç kaldım.",
        "Her şey çok güzeldi, kesinlikle tavsiye ederim.",
        "Yemekler yağlıydı ama lezzetliydi.",
        "Rezervasyonum olmasına rağmen 20 dakika bekledim.",
        "Güler yüzlü personel, temiz ortam.",
        "Yemeklerin tadı mükemmeldi, tekrar geleceğim.",
        "Sipariş çok geç geldi, soğumuştu.",
        "Ambiyans çok hoştu, romantik bir akşam geçirdik.",
        "Garsonlar kaba davranıyordu.",
        "Yemek çeşitliliği çok iyiydi.",
        "Sandalyeler rahatsızdı, keyif alamadım.",
        "Kebap çok lezzetliydi, sıcak geldi.",
        "Fiyatlar aşırı pahalı, lezzet de ortalama.",
        "Manzara mükemmeldi, yemekler de öyle.",
        "Et pişmemişti, midem bozuldu.",
        "Tatlılar taze ve enfesti.",
        "Masamız geç hazırlandı, kötü başlangıç.",
        "Lezzetliydi ama beklemeye değmezdi.",
        "Personel çok ilgiliydi, çocuklara oyuncak verdiler.",
        "Hizmet kalitesi çok düşüktü.",
        "Yemekler hızlı geldi, sıcak ve tazeydi.",
        "Açık büfe bayattı, hiç memnun kalmadım.",
        "Kahvaltı tabağı oldukça zengindi.",
        "Bir kahveye 80 TL verdik, çok saçma.",
        "Rezervasyonumuzu kaydetmemişler.",
        "İç dekorasyon çok şıktı.",
        "Garson siparişleri karıştırdı.",
        "Fiyat performans açısından mükemmel.",
        "Yemek soğuktu, ilgilenen olmadı.",
        "Tat olarak vasattı, beklentimi karşılamadı.",
        "Patates kızartması çıtır çıtırdı.",
        "Müşteri memnuniyeti sıfır.",
        "Yemekler çok geç geldi, üstelik soğuktu.",
        "Makarna sosu çok güzeldi.",
        "İlgili çalışanlar ve temiz bir mekan.",
        "Tatlıların sunumu muhteşemdi.",
        "Sipariş verdiğim yemek başka bir şeydi.",
        "Bahşiş istemeleri hoşuma gitmedi.",
        "Aile ortamı çok hoştu.",
        "Çorba sanki dünden kalmaydı.",
        "Hızlı servis ve güzel sunum.",
        "Mekanda sigara içiliyordu, rahatsız oldum.",
        "Kahve yanında lokum getirmeleri hoş detaydı.",
        "Aşçının özel yemeği çok lezzetliydi.",
        "İlk ve son gelişim.",
        "Servis personeli çok saygılıydı.",
        "Yemekler doyurucu ve kaliteliydi.",
        "İndirim olmasına rağmen kaliteden ödün verilmemişti.",
        "İçecekler eksik geldi.",
        "Kebaplar tam kıvamındaydı.",
        "Çok gürültülü, dinlenemedik.",
        "Manzaraya karşı çay içmek harikaydı.",
        "Çalışanlar oldukça ilgisizdi.",
        "Tatlıdan sinek çıktı.",
        "Restoran çok temizdi.",
        "Yemek siparişi karıştı ama telafi ettiler.",
        "Fiyatlar çok uygun.",
        "Sadece manzara için gidilir.",
        "Müşteriyle ilgilenmiyorlar.",
        "Tatlı menüsü çok başarılıydı.",
        "Sınırlı menü, alternatif yoktu.",
        "İkramlar çok cömertti.",
        "Kötü servis, lezzetsiz yemekler.",
        "Sunum harikaydı, tadı da öyle.",
        "Masalar çok sıkışık yerleştirilmişti.",
        "Yemekten sonra ikram edilen çay çok güzeldi.",
        "Aşırı yağlıydı, mide bulandırıcı.",
        "Çocuklar için oyun alanı olması büyük avantaj.",
        "Sadece tatlılar güzeldi, gerisi vasat.",
        "Çalışanlar çok cana yakındı.",
        "Tabağımda saç çıktı!",
        "Ambiyans harikaydı.",
        "Çok bekledik ama değdi.",
        "Bekleme süresi aşırı uzundu.",
        "Her şey taze ve lezzetliydi.",
        "Servis yavaştı ama yemekler telafi etti.",
        "Gözleme çok lezzetliydi.",
        "Garsonlar saygılıydı ama yemekler kötüydü.",
        "İlk defa gittim, çok memnun kaldım.",
        "Buz gibi yemek geldi.",
        "Lezzet şahane ama porsiyon küçük.",
        "Yemek sonrası ikram hoş bir sürprizdi.",
        "Etin içi çiğdi.",
        "Güleryüzlü hizmet ve nefis tatlar.",
        "İçecekler çok pahalıydı.",
        "Düğünümüzü burada yaptık, her şey mükemmeldi.",
        "Soğuk çay verdiler.",
        "Tavuk çok güzel pişmişti.",
        "Sipariş yanlış geldi, düzeltmediler.",
        "Kahvaltısı çok zayıftı.",
        "Personel çocuklarla çok ilgilendi.",
        "Kalabalık ama organizasyon iyiydi.",
        "Koku çok kötüydü.",
        "Hızlı ve sıcak servis.",
        "Güzel manzara, kötü yemek.",
        "Tatlının yanında kahve ikramı çok hoştu.",
        "Bir daha gelmem, memnun kalmadım.",
        "Deniz ürünleri çok tazeydi.",
        "Garsonlar masamızı sürekli unuttu.",
        "Pizzası mükemmeldi.",
        "Siparişimiz eksik geldi ama hemen hallettiler.",
        "İçecekler buz gibiydi, çok serinleticiydi.",
        "Kuru ve tatsızdı, beğenmedim."
    ][:100],  # 👈 This trims it to 100 exactly
    "label": [
        "positive", "negative", "neutral", "negative", "positive", "negative", "positive", "neutral", "negative", "positive",
        "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "neutral", "positive", "negative", "positive", "negative", "positive", "negative", "negative",
        "positive", "negative", "negative", "positive", "negative", "negative", "negative", "positive", "positive", "positive",
        "negative", "negative", "positive", "negative", "positive", "negative", "positive", "positive", "negative", "positive",
        "positive", "positive", "negative", "positive", "negative", "positive", "negative", "negative", "positive", "neutral",
        "negative", "positive", "negative", "positive", "negative", "positive", "negative", "positive", "positive", "positive",
        "negative", "positive", "neutral", "positive", "negative", "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative", "positive", "positive", "negative", "positive", "negative"
    ]
}

# ✅ Check that both lengths match
print(f"Number of texts: {len(data['text'])}")
print(f"Number of labels: {len(data['label'])}")

#%%
# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.")
# %%
# pad the sequences
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
print(padded_sequences.shape)

# %%
Label_encoder = LabelEncoder()
labels = Label_encoder.fit_transform(data['label'])
print(f"Encoded labels: {labels}")
# %%
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
# %%

sentences = [text.split() for text in data['text']]
# train a Word2Vec model
word2vec_model = Word2Vec(sentences, vector_size=50, window=5 , min_count=1, workers=4)

# create an embedding matrix
emmeding_matrix = np.zeros((len(word_index) + 1, 50))
for word, i in word_index.items():
    emmeding_vector = word2vec_model.wv[word] if word in word2vec_model.wv else None
    if emmeding_vector is not None:
        emmeding_matrix[i] = emmeding_vector

# %%
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=50,
                  weights=[emmeding_matrix], input_length=max_length, trainable=False))
model.add(SimpleRNN(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(SimpleRNN(32))
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))  # 3 classes: positive, negative, neutral
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=3, validation_data=(X_test, y_test), verbose=1) 

# %%
model.summary()
# %%
# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# %%
# make predictions
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=1)
    return Label_encoder.inverse_transform(predicted_label)[0]

# Test the prediction function
test_texts = [
    "Yemekler çok lezzetliydi, tekrar geleceğim.",
    "Garson çok ilgisizdi, bir daha gelmem.",
    "Tatlı mükemmeldi ama ana yemek soğuktu."]
for text in test_texts:
    sentiment = predict_sentiment(text)
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")
# %%
