# 犬と猫の画像分類アプリ

## 概要
- 犬または猫の画像をアップロードすると、どちらか判断します。

## 動作環境
Python 3.6.1
TensorFlow 2.0.0

## 制作理由
- 元々は、TECH::EXPERTのカリキュラムで<a href="https://github.com/sou0913/freemarket_sample_61a">メルカリのクローンサイト<a>の作成に取り組んだ際、商品のカテゴリーを自動で分類できるようにしたいと思いTensorFlowを学習しました。
 - しかし、学習が追いつかず、機能の実装は断念しました。代わりにTensorFlowの勉強のために作ったのがこちらです。

## 使用技術
- Googleの機械学習ライブラリ、Tensorflowを用いました。
- Pythonのフレームワーク、Flaskでインターフェイスを実装しました。

## 制作上のポイント
- 機械学習に使う画像は、GoogleChromeからスクレイピングをして集めました。（勉強用などに用意されているデータセットを使いませんでした）

<a href="https://gyazo.com/dd2f61596dd87d0cfb72b4bf83f8db35"><img src="https://i.gyazo.com/dd2f61596dd87d0cfb72b4bf83f8db35.gif" alt="Image from Gyazo" width="400"/></a>

- 正答率がなかなか上がらなかったので、事前に訓練されたモデル(MobileNet V2)を利用する転移学習を取り入れました。その結果、テスト画像で95％を超える正解率を達成しました。

<a href="https://gyazo.com/e7b1bd67eabbbac60ab2841d3133943f"><img src="https://i.gyazo.com/e7b1bd67eabbbac60ab2841d3133943f.gif" alt="Image from Gyazo" width="400"/></a>
 
## 制作で学んだこと
- TensorFlowを利用した画像分類の一通りの流れを理解し、実践できるようになりました。（素材集めからモデル構築、外部モデルの利用、検証など）
- Flaskを利用した簡単なアプリケーション作成の流れを把握できました。
