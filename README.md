# textEmotion
テキストボックスに入力した英文をpositiveかnegativeか分類するWebアプリケーションです。
IDBM映画レビューデータセットを用いて機械学習を行いました。
再配布可能か判断できなかったため、データセットは載せていません。

# 動かすためにダウンロードしなければならないファイル一覧とその説明
- templates/index.html
  - テキストの入力ページの作成
- templatex/result.html
  - 感情分類した結果を表示するページの作成
- svc_model.pkl
  - 学習した非線形サポートベクターマシンのモデルを保存したもの
- svc_vectorizer.pkl
  - 学習に使用したtf-idfベクトル変換器
- textEmotion.py
  - 実際にWebアプリケーションを起動して、テキスト感情分類を行うPython3プログラム

# 使い方
1. 上記のダウンロードしなければならないファイルを、ダウンロードしてください。
2. textEmotion.pyを実行(「python3 textEmotion.py」で実行できます。)して、Webアプリケーションを開いてください。
3. テキストボックスに英文を入力してください。
4. 「分類する」ボタンを押してください。
5. 分類結果が、positiveもしくはnegativeで表示されます。

# ダウンロードしなくても良いファイルの説明
- label_test.pkl
  - テスト用のラベルデータを保存したファイル
- label_train.pkl
  - 学習用のラベルデータを保存したファイル
- learnModel.py
  - モデルを学習に使用したPython3プログラム
- processText.py
  - レビューデータの前処理に使用したPython3プログラム
- text_test.pkl
  - テスト用のレビューデータを保存したファイル
- text_train.pkl
  - 学習用のレビューデータを保存したファイル
