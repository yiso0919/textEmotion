# テキストに対する前処理を行い、その結果を保存するプログラム

# データ分析のためのライブラリ
import pandas as pd
# データを学習データとテストデータに分割するためにインポート
from sklearn.model_selection import train_test_split
# 正規表現をインポート
import re
# ストップワードをインポート
from nltk.corpus import stopwords
# 自然言語処理のためのライブラリをインポート
import nltk
# 見出し語かのためのインポート
from nltk.stem.wordnet import WordNetLemmatizer as WNL
# 保存するためのpickleをインポート
import pickle
# ストップワードをダウンロード
nltk.download('stopwords')

# IMDB Dataset.csvのレビュー情報を読み込む
text_data = pd.read_csv('IMDB Dataset.csv', usecols=[0])
# IMDB Dataset.csvのラベル情報を読み込む
label_data = pd.read_csv('IMDB Dataset.csv', usecols=[1])
# 50000件のレビューセットを学習データ25000件:テストデータ25000件のバランスで分割
text_train, text_test, label_train, label_test = train_test_split(text_data, label_data, test_size = 0.5, shuffle = False)
# 25000回繰り返す
for i in range(25000):
    # text_trainのi番目の要素のHTMLタグを除去する
    text_train.at[i, 'review'] = re.sub(re.compile('<.*?>'), ' ', text_train.at[i, 'review'])
    # text_testのi番目の要素のHTMLタグを除去する
    text_test.at[i+25000, 'review'] = re.sub(re.compile('<.*?>'), ' ', text_test.at[i+25000, 'review'])
    # text_trainのi番目の要素のアルファベット、「'」、「-」以外の文字を除去する
    text_train.at[i, 'review'] = re.sub(r'[^a-zA-Z\'\-]', ' ', text_train.at[i, 'review'])
    # text_testのi番目の要素のアルファベット、「'」、「-」以外の文字を除去する
    text_test.at[i+25000, 'review'] = re.sub(r'[^a-zA-Z\'\-]', ' ', text_test.at[i+25000, 'review'])
    # text_train内の大文字を小文字に統一する
    text_train.at[i, 'review'] = text_train.at[i, 'review'].lower()
    # text_test内の大文字を小文字に統一する
    text_test.at[i+25000, 'review'] = text_test.at[i+25000, 'review'].lower()
# 25000回繰り返す
for i in range(25000):
    # text_trainを単語ごとに分割する
    text_train.at[i, 'review'] = text_train.at[i, 'review'].split()
    # text_testを単語ごとに分割する
    text_test.at[i+25000, 'review'] = text_test.at[i+25000, 'review'].split()
# 見出し語化のためのWNLオブジェクトを作成
wnl = WNL()
# 25000回繰り返す
for i in range(25000):
    # リストの要素を削除した時のズレを補正するためのカウンタ
    k=0
    # i番目のレビューに含まれる単語の数だけ繰り返す
    for j in range(len(text_train.at[i, 'review'])):
        # 単語ごとに「'」もしくは「-」もしくはストップワードが含まれているかどうか判定
        if '\'' in text_train.at[i, 'review'][j-k] or '-' in text_train.at[i, 'review'][j-k] or text_train.at[i, 'review'][j-k] in stopwords.words('english'):
            # 含まれている場合は、その単語をリストから削除する
            text_train.at[i, 'review'].remove(text_train.at[i, 'review'][j-k])
            # リストの要素を削除した時のズレを補正するために増加
            k += 1
        # 除去されない単語の場合
        else:
            # wnlを用いて一般的な形にする
            text_train.at[i, 'review'][j-k] = wnl.lemmatize(text_train.at[i, 'review'][j-k])
    # リストの要素を削除した時のズレを補正するためのカウンタ
    l=0
    # i番目のレビューに含まれる単語の数だけ繰り返す
    for m in range(len(text_test.at[i+25000, 'review'])):
        # 単語ごとに「'」もしくは「-」もしくはストップワードが含まれているかどうか判定
        if '\'' in text_test.at[i+25000, 'review'][m-l] or '-' in text_test.at[i+25000, 'review'][m-l] or text_test.at[i+25000, 'review'][m-l] in stopwords.words('english'):
            # 含まれている場合は、その単語をリストから削除する
            text_test.at[i+25000, 'review'].remove(text_test.at[i+25000, 'review'][m-l])
            # リストの要素を削除した時のズレを補正するために増加
            l += 1
        # 除去されない単語の場合
        else:
            # wnlを用いて一般的な形にする
            text_test.at[i+25000, 'review'][m-l] = wnl.lemmatize(text_test.at[i+25000, 'review'][m-l])
# 25000回繰り返す
for i in range(25000):
    # 単語ごとに分割されたtext_trainをレビューごとの単位にまとめる
    text_train.at[i, 'review'] = ' '.join(text_train.at[i, 'review'])
    # 単語ごとに分割されたtext_testをレビューごとの単位にまとめる
    text_test.at[i+25000, 'review'] = ' '.join(text_test.at[i+25000, 'review'])
# text_trainを保存するためのtext_train.pklを開く
with open('text_train.pkl', 'wb') as f1:
    # text_trainを保存する
    pickle.dump((text_train), f1)
# text_testを保存するためのtext_test.pklを開く
with open('text_test.pkl', 'wb') as f2:
    # text_testを保存する
    pickle.dump((text_test), f2)
# label_trainを保存するためのlabel_train.pklを開く
with open('label_train.pkl', 'wb') as f3:
    # label_trainを保存する
    pickle.dump((label_train), f3)
# label_testを保存するためのlabel_test.pklを開く
with open('label_test.pkl', 'wb') as f4:
    # label_testを保存する
    pickle.dump((label_test), f4)