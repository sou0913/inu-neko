# 犬と猫の画像分類アプリ

## 概要
- 犬または猫の画像をアップロードすると、どちらか判断します。

## 使用技術
- Googleの機械学習ライブラリ、Tensorflowを用いました。
- Pythonの軽量ライブラリ、Flaskでインターフェイスを実装しました。

## 制作上のポイント
- 機械学習に使う画像は、GoogleChromeからスクレイピングをして集めました。（勉強用などに用意されているデータセットを使いませんでした）
- 正答率がなかなか上がらなかったので、事前に訓練されたモデル(MobileNet V2)を利用する転移学習を取り入れました。その結果、テスト画像で95％を超える正解率を達成しました。
 
## 制作で学んだこと
- TensorFlowを利用した画像分類の一通りの流れを理解し、実践できるようになりました。（素材集めからモデル構築、外部モデルの利用、検証など）
- Flaskを利用した簡単なアプリケーション作成の流れを把握できました。
