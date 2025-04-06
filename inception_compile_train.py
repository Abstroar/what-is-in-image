from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import add, LSTM, Embedding, Dense, Dropout, Input
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint

def token_dic_loader():
    with open("tokenizer_1.json", "r", encoding="utf-8") as f:
        word_index = json.load(f)

    tok = Tokenizer(num_words=5000, oov_token="<unk>")
    tok.word_index = word_index
    tok.index_word = {i: w for w, i in word_index.items()}
    return tok

def model_creator(max_length, vocabulary_size, embedding_dim):

    inputs1 = Input(shape=(4096,))
    features1 = Dropout(0.5)(inputs1)
    features2 = Dense(256, activation='relu')(features1)
    inputs2 = Input(shape=(max_length,))
    seqfeatures1 = Embedding(vocabulary_size, embedding_dim, mask_zero=True)(inputs2)
    seqfeatures2 = Dropout(0.5)(seqfeatures1)
    seqfeatures3 = LSTM(256)(seqfeatures2)

    decoder1 = add([features2, seqfeatures3])

    decoder2 = Dense(256, activation='relu')(decoder1)

    outputs = Dense(vocabulary_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

class CaptionDataGenerator(Sequence):
    def __init__(self, data_dict, tokenizer, max_length, vocab_size, batch_size=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.data_items = [(caption, np.array(feature, dtype=np.float32)) for caption, feature in data_dict.items()]
        np.random.shuffle(self.data_items)

    def __len__(self):
        return int(np.ceil(len(self.data_items) / self.batch_size))

    def __getitem__(self, idx):
        batch_items = self.data_items[idx * self.batch_size:(idx + 1) * self.batch_size]
        X1, X2, y = [], [], []

        for caption, feature in batch_items:
            seq = self.tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_word = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]

                X1.append(feature)
                X2.append(in_seq)
                y.append(out_word)

        return [np.array(X1), np.array(X2)], np.array(y)

    def on_epoch_end(self):
        np.random.shuffle(self.data_items)

if __name__ == "__main__":

    feature_file = "corr_feature.pkl"
    max_length = 20
    vocabulary_size = 5000
    embedding_dim = 200
    batch_size = 32
    # token_dictionary_creator()
    tokenizer = token_dic_loader()


    model = model_creator(max_length, vocabulary_size, embedding_dim)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    with open("features.json","r") as data:
        dict = json.load(data)
        model.summary()
        generator = CaptionDataGenerator(dict, tokenizer, max_length, vocabulary_size, batch_size=32)
        checkpoint = ModelCheckpoint(
            filepath='model_checkpoint_{epoch:02d}.keras',
            save_best_only=False,
            save_weights_only=False,
            verbose=1
        )
        model.fit(generator, epochs=20, callbacks=[checkpoint])




