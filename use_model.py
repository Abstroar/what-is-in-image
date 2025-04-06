from extra_feature import extract_feature
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import add, LSTM, Embedding, Dense, Dropout, Input
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import random

def token_dict_loader():
    with open("tokenizer_1.json", "r", encoding="utf-8") as f:
        word_index = json.load(f)

    tok = Tokenizer(num_words=5000, oov_token="<unk>")
    tok.word_index = word_index
    tok.index_word = {i: w for w, i in word_index.items()}
    return tok

def generate_caption(model, tokenizer, photo_feature, max_length):
    in_text = "startseq"
    print("Starting caption generation...")

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        photo_feature = np.array(photo_feature, dtype=np.float32)
        if photo_feature.ndim == 2:
            photo_feature = photo_feature[0]
        photo_feature = photo_feature.reshape((1, 4096))

        yhat = model.predict([photo_feature, sequence], verbose=0)
        print("yhat",yhat)
        print("Top 5 predictions:", np.argsort(yhat[0])[-5:][::-1])
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)

        print(f"[Step {i + 1}] yhat_index: {yhat_index}, word: {word}")

        if word is None:
            print("Prediction returned None. Stopping.")
            word = "nonworking"

        in_text += ' ' + word

        if word == 'endseq':
            print("End sequence token encountered. Stopping.")
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()


if __name__ == "__main__":
    tokeni = token_dict_loader()
    model = load_model("model_checkpoint_20.keras")
    test = "test/front-view-friends-dinner-party.jpg"
    featuree = extract_feature(test)
    print(test)
    ok = generate_caption(model, tokeni, featuree, 20)
    print(ok)