# classification-keras
Kerasライブラリを使用した画像分類のサンプルです。

# 新たなデータセットの作成
datasetsフォルダにサンプル画像を置いています。<br>
このフォルダ構成を参考に、新たなデータセットを作成することができます。<br>
１．分類したいクラス分のフォルダを、trainとtestフォルダ内にそれぞれ作成してください。<br>
２．作成したクラスフォルダに画像を配置してください。<br>

# 新たなネットワークモデルの作成
train.pyのsample_model関数にサンプルモデルがあります。<br>
ここを参考にモデルを書き換えることができます。<br>

# 学習処理
train.pyのbatch_sizeとepochs<br>
fit_generatorのsteps_per_epochとvalidation_steps<br>
を適切に変更してください。
```
python train.py
```

# 推論処理
path = 'datasets/test/cat/1.jpg'を推論させたい画像に変更してください。
```
python test.py
```
