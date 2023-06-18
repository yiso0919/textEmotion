# 前処理したテキストデータからモデルを作成し保存するプログラム

# tf-idfベクトルに変換するためにインポート
from sklearn.feature_extraction.text import TfidfVectorizer
# pklファイルのオープン、保存のためにインポート
import pickle
# SVCをインポート
from sklearn.svm import SVC
# 正解率を表示するためにインポート
from sklearn.metrics import accuracy_score

# 学習用テキストデータが保存されたファイルを開く
with open('text_train.pkl', "rb") as f1:
    # 学習用テキストデータをロードする
    text_train = pickle.load(f1)
# テスト用テキストデータが保存されたファイルを開く
with open('text_test.pkl', "rb") as f2:
    # テスト用テキストデータをロードする
    text_test = pickle.load(f2)
# 学習用ラベルデータが保存されたファイルを開く
with open('label_train.pkl', "rb") as f3:
    # 学習用ラベルデータをロードする
    label_train = pickle.load(f3)
# テスト用ラベルデータが保存されたファイルを開く
with open('label_test.pkl', "rb") as f4:
    # テスト用ラベルデータをロードする
    label_test = pickle.load(f4)
# tf-idfベクトル化のためのオブジェクトを作成
vectorizer = TfidfVectorizer()
# 学習用テキストデータをtf-idfベクトル化
train_features = vectorizer.fit_transform(text_train['review'])
# テスト用テキストデータをtf-idfベクトル化
test_features = vectorizer.transform(text_test['review'])
# SVCのモデルを作成
svc_model = SVC(verbose=True)
# SVCを学習させる
svc_model.fit(train_features, label_train['sentiment'])
# 学習用データを用いて予測
pred_train = svc_model.predict(train_features)
# 予測の精度を表示
print(accuracy_score(label_train['sentiment'], pred_train)) # 0.99136
# テスト用データを用いて予測
pred_test = svc_model.predict(test_features)
# 予測の精度を表示
print(accuracy_score(label_test['sentiment'], pred_test)) # 0.89216

# SVCのモデルを保存するためのpklファイルを開く
with open('svc_model.pkl', 'wb') as file1:
    # 実際に保存する
    pickle.dump((svc_model), file1)
# tf-idfベクトル変換器を保存するためのpklファイルを開く
with open('svc_vectorizer.pkl', 'wb') as file2:
    # 実際に保存する
    pickle.dump((vectorizer), file2)