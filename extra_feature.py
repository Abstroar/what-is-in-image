from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
from keras.applications.vgg16 import VGG16, preprocess_input
import pickle

img_paths = "./img_to_txt_dataset/images"
cap_path = "./img_to_txt_dataset/captions.txt"

full_model = VGG16(weights='imagenet', include_top=True)
last_needed_layer = full_model.get_layer('fc2')
model = Model(full_model.input, last_needed_layer.output)

def extract_feature(img):
    img = load_img(img, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature = model.predict(img)
    feature = np.reshape(feature, feature.shape[1])
    print(feature[0:10])
    return feature

if __name__=="__main__":
    dictionary = {}
    with open(cap_path, 'r') as captions:
        x = 0
        for i in captions:
            x+=1
            image, cap = i.split(',',1)
            cap = cap.strip().replace("\n", "")
            ok = f"startseq {cap} endseq"
            dictionary[ok] = extract_feature(img_paths+"/"+image).tolist()
    # with open("features.pkl", "wb") as f:
    #     pickle.dump(dictionary, f)
    with open("features.json", "w") as f:
        json.dump(dictionary, f, indent=4)
