"""
画像分類のためのニューラルネットワーク学習プログラムです。
まずは、trainとtest用に各クラスのフォルダを作成して、そこに画像を配置してください。
"""
from keras import models
import numpy as np
import json
from keras.preprocessing.image import load_img, img_to_array

if __name__ == '__main__':
    height = 256
    wight = 256
    classes = 3

    path = 'datasets/test/cat/1.jpg'
    img = load_img(path, grayscale=False, color_mode='rgb', target_size=(height, wight), interpolation='bilinear')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) # 次元を(256, 256, 3)から(1, 256, 256, 3)に変更
    img /= 255.0

    labels = [''] * classes
    with open('labels.json', 'r') as f:
        tmp = json.load(f)
        for k, v in tmp.items():
            labels[v] = k

    model = models.load_model('model_weights.h5')

    output = model.predict(img)

    print(output)

    pred = np.argmax(output, axis = 1) # 値が一番大きなインデックスを取得する = 認識されたクラスインデックス

    print('この画像は、「' + labels[pred[0]] + '」と認識されました。')






