"""
画像分類のためのニューラルネットワーク学習プログラムです。
まずは、trainとtest用に各クラスのフォルダを作成して、そこに画像を配置してください。
"""
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, Dense
from keras.preprocessing.image import ImageDataGenerator
import json

def sample_model(input_shape, classes):
    inputs = Input(shape=input_shape)

    x = conv3_bn_relu(inputs, 64, '1')
    x = conv3_bn_relu(inputs, 64, '2')
    x = MaxPooling2D()(x)

    x = conv3_bn_relu(inputs, 128, '3')
    x = conv3_bn_relu(inputs, 128, '4')
    x = MaxPooling2D()(x)

    x = conv3_bn_relu(inputs, 256, '5')
    x = conv3_bn_relu(inputs, 256, '6')
    x = MaxPooling2D()(x)

    x = conv3_bn_relu(inputs, 512, '7')

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax', kernel_initializer='he_normal')(x)

    return Model(inputs, x)

def conv3_bn_relu(x, filters, suffix):
    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', name='conv3' + suffix)(x)
    x = BatchNormalization(name='conv3_bn' + suffix)(x)
    x = Activation('relu')(x)

    return x

if __name__ == '__main__':
    batch_size = 1
    epochs = 1
    height = 256
    wight = 256
    classes = 3

    train_datagen = ImageDataGenerator(
        # 前処理　画素値を0～1の範囲に正規化
        rescale=1.0 / 255,
        # データ拡張
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        'datasets/train',
        target_size=(256, 256),
        batch_size=batch_size,
        # クラス分類の場合はcategoricalを指定する
        class_mode='categorical',
        shuffle=True)
    labels = train_generator.class_indices
    with open('labels.json', 'w') as f:
        json.dump(labels, f)

    test_generator = test_datagen.flow_from_directory(
        'datasets/test',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    model = sample_model((height, wight, 3), classes)

    model.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=21, #教師データ数 / バッチ数
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=6) #評価データ数 / バッチ数

    model.save('model_weights.h5')
    with open('history.json', 'w') as f:
        json.dump(history.history, f)