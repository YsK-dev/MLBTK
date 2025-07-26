# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer

# %%
data = {
    "text": [
        "Motor sürmek tüm sıkıntıları çözüyor",
        "Kitap okumak aşırı bilgilendirici",
        "Kahve içmek güne başlamak için harika",
        "Yürüyüş yapmak zihni dinlendiriyor",
        "Müzik dinlemek ruhuma iyi geliyor",
        "Film izlemek başka hayatlara dokunmak gibi",
        "Gitar çalmak beni benden alıyor",
        "Yoga yapmak iç huzuru sağlıyor",
        "Kod yazmak üretici hissettiriyor",
        "Resim yapmak terapi gibi geliyor",
        "Şiir okumak duyguları yoğunlaştırıyor",
        "Bisiklet sürmek özgürlük gibi",
        "Doğa gezileri ruhumu yeniliyor",
        "Meditasyon yapmak farkındalık kazandırıyor",
        "Pasta yapmak mutlu hissettiriyor",
        "Kamp yapmak doğayla bütünleşmek demek",
        "Denize girmek tüm stresi alıyor",
        "Fotoğraf çekmek anı ölümsüzleştiriyor",
        "Arkadaşlarla konuşmak moral veriyor",
        "Kedi sevmek huzur veriyor",
        "Dans etmek enerjiyi yükseltiyor",
        "Sabah erken kalkmak verimliliği artırıyor",
        "Yabancı dil öğrenmek vizyon kazandırıyor",
        "Yüzmek vücudu rahatlatıyor",
        "Satranç oynamak zihni keskinleştiriyor",
        "Bahçe işleri yapmak doğayla bağ kurduruyor",
        "Kitap yazmak içsel yolculuk gibi",
        "Puzzle yapmak odaklanmayı geliştiriyor",
        "Tiyatro izlemek empati kurduruyor",
        "Yeni tarifler denemek yaratıcılığı artırıyor",
        "Koşmak bedeni zinde tutuyor",
        "Günlük yazmak kendini tanımaya yardımcı",
        "Spor salonuna gitmek disiplini öğretiyor",
        "Podcast dinlemek farklı bakışlar kazandırıyor",
        "Hayal kurmak motivasyon veriyor",
        "Göl kenarında oturmak huzur veriyor",
        "Tek başına sinemaya gitmek cesaret verici",
        "Gönüllü çalışmak anlam katıyor",
        "Tarih okumak geçmişi anlamayı sağlıyor",
        "Borsa takip etmek strateji geliştiriyor",
        "Mobil oyunlar oynamak eğlenceli kaçış sunuyor",
        "Film analizi yapmak düşünmeyi geliştiriyor",
        "El işi yapmak dikkat geliştiriyor",
        "Kamp ateşi başında oturmak büyüleyici",
        "Kuş sesleri dinlemek rahatlatıcı",
        "Serbest yazı yazmak yaratıcı hissettiriyor",
        "Kaligrafi yapmak estetik duyguyu geliştiriyor",
        "Dağcılık yapmak cesaret kazandırıyor",
        "Köpek gezdirmek günlük rutini güzelleştiriyor",
        "Çiçek yetiştirmek sabrı öğretiyor",
        "Ağaçlara sarılmak doğayı hissettiriyor",
        "Dalgaları dinlemek iç huzur veriyor",
        "Gündoğumu izlemek umut veriyor",
        "Aileyle vakit geçirmek bağları güçlendiriyor",
        "Yeni yerler keşfetmek heyecan verici",
        "Kamp yaparken yıldızları izlemek büyüleyici",
        "Tırmanış yapmak sınırları zorluyor",
        "Kar yağışını izlemek içsel sessizlik sunuyor",
        "Hikaye yazmak duyguları dışa vuruyor",
        "Blog tutmak düşünceleri netleştiriyor",
        "Sokak sanatı görmek ilham veriyor",
        "Plajda yürümek dinlendirici",
        "Yeni insanlar tanımak ufku genişletiyor",
        "Eski eşyaları tamir etmek tatmin edici",
        "Kumda çıplak ayak yürümek rahatlatıcı",
        "Müzik yapmak içsel ifade sağlıyor",
        "Sanat galerisi gezmek hayal gücünü besliyor",
        "Sabah kahvaltısı hazırlamak güne anlam katıyor",
        "Kalabalıktan uzaklaşmak zihni tazeliyor",
        "Yalnız kalmak düşünmeye zaman veriyor",
        "Sokak hayvanlarına yardım etmek mutluluk veriyor",
        "Eski dostlarla buluşmak nostaljik",
        "Karakalem çalışmak odaklanmayı sağlıyor",
        "Röportaj izlemek farklı hayatları tanıtıyor",
        "Kendi kendine konuşmak farkındalık kazandırıyor",
        "Oyun oynamak zihinsel egzersiz gibi",
        "Sosyal medyadan uzak durmak zihni temizliyor",
        "Kütüphanede vakit geçirmek dinginleştirici",
        "Yıldızlara bakmak sonsuzluk hissi veriyor",
        "Bitki çayı içmek sakinleştirici",
        "Ev düzenlemek zihin düzenliyor",
        "Gölge oyunları izlemek hayal gücünü çalıştırıyor",
        "Birine mektup yazmak samimiyet kazandırıyor",
        "Sokakta yürürken müzik dinlemek sinematik",
        "Piknik yapmak doğayla buluşma anı",
        "Tek başına kahve içmek kendini dinlemek gibi",
        "Felsefe kitapları okumak bakış açısını değiştiriyor",
        "Astronomiyle ilgilenmek evreni sorgulatıyor",
        "Bir film üzerine düşünmek farkındalık sağlıyor",
        "Bir projeyi tamamlamak tatmin verici",
        "El yapımı hediye hazırlamak anlamlı",
        "Sade kahve içmek gerçeklerle yüzleşmek gibi",
        "Fotoğraf albümü düzenlemek anılarla buluşmak",
        "Sokakta sessizce yürümek içe dönüş sağlıyor",
        "Şarkı sözü yazmak iç dünyayı dışa vurmak",
        "Bilinçli nefes almak kontrol hissi veriyor",
        "Tek başına seyahat etmek kişisel gelişim demek",
        "Yeni bir beceri öğrenmek özgüveni artırıyor",
        "Birine yardım etmek insanı iyi hissettiriyor",
        "Kendi işini yapmak özgürlük veriyor",
        "Günü planlamak verim kazandırıyor",
        "Ebeveynlerle vakit geçirmek duygusal bağ kurduruyor",
        "Buhar banyosu yapmak rahatlatıcı",
        "Gece yürüyüşü yapmak dinginlik sağlıyor"
    ]
}
print("number of samples:", len(data["text"]))
# %%
# text preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data["text"]) # learns word frequencies
total_words = len(tokenizer.word_index) + 1
print("total words:", total_words)
# %%
# create n-grams and padding
input_sequences = []
for line in data["text"]:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)
print("number of input sequences:", len(input_sequences))
max_sequence_length = max([len(x) for x in input_sequences])
print("max sequence length:", max_sequence_length)
# %%
# find the max sequence length, pad the sequences
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

print("padded input sequences shape:", input_sequences.shape)

# x (input) and y (output) split
xs, ys = input_sequences[:, :-1], input_sequences[:, -1]
print("x shape:", xs.shape, "y shape:", ys.shape)

# %%
# one-hot encode the output
ys = tf.keras.utils.to_categorical(ys, num_classes=total_words)
print("one-hot encoded y shape:", ys.shape)
# %%
# LSTM model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_length - 1))

model.add(LSTM(100 , return_sequences=False))

#output layer
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
history = model.fit(xs, ys, epochs=100, verbose=1)

#show the model summary
model.summary()

# %%
def generate_text(seed_text, next_words, max_sequence_length):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        predicted_prob = predicted[0][predicted_word_index]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        print(f"Predicted word: '{output_word}' ({predicted_prob*100:.2f}%)")
        seed_text += " " + output_word
    return seed_text


# %%
# generate text
seed_text = "Kitap okumak"
next_words = 7
generated_text = generate_text(seed_text, next_words, max_sequence_length)
print("Generated text:", generated_text)
# %%

