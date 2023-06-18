# 与えられたテキストがpositiveかnegativeのどちらか分類する

# Webフレームワーク
from flask import Flask, render_template, request
# 自然言語処理ライブラリ
import nltk
# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
# pklファイルを開けるため
import pickle
# 正規表現
import re
# ストップワードをインポート
from nltk.corpus import stopwords
# 見出し語化のため
from nltk.stem.wordnet import WordNetLemmatizer as WNL

# 必要なデータのダウンロード
nltk.download('punkt')
# ストップワードをダウンロード
nltk.download('stopwords')

# Flaskアプリケーションの作成
app = Flask(__name__)

# ルートページ
@app.route('/')
def index():
    return render_template('index.html')

# 予測結果の表示ページ
@app.route('/predict', methods=['POST'])
def predict():
    # テキストの取得
    test_text = request.form['text']

    # HTMLタグを除去
    test_text = re.sub(re.compile('<.*?>'), ' ', test_text)
    # アルファベットと「'」、「-」以外の文字を除去
    test_text = re.sub(r'[^a-zA-Z\'\-]', ' ', test_text)
    # 大文字を小文字に統一
    test_text = test_text.lower()
    # 単語ごとに分割
    test_text = test_text.split()
    # 見出し語化のためのWNLオブジェクトを生成
    wnl = WNL()
    # 要素を削除した時のズレを補正するためのカウンタ
    count = 0
    # test_textの単語数だけ繰り返す
    for i in range(len(test_text)):
        # 単語ごとに「'」もしくは「-」もしくはストップワードが含まれているかどうか判定
        if '\'' in test_text[i-count] or '-' in test_text[i-count] or test_text[i-count] in stopwords.words('english'):
            # 含まれている場合は、その単語をリストから削除する
            test_text.remove(test_text[i-count])
            # リストの要素を削除した時のズレを補正するために増加
            count += 1
        # 除去されない単語の場合
        else:
            # wnlを用いて一般的な形にする
            test_text[i-count] = wnl.lemmatize(test_text[i-count])
    # 分割した単語をまとめる
    test_text = ' '.join(test_text)

    # 学習済みのSVCモデルをオープン
    with open('svc_model.pkl', 'rb') as file:
        # SVCモデルをロード
        classifier = pickle.load(file)
    # 学習済みのtf-idf変換器をオープン    
    with open('svc_vectorizer.pkl', 'rb') as file:
        # th-idf変換器をロード
        vectorizer = pickle.load(file)

    # 特徴ベクトル化
    test_features = vectorizer.transform([test_text])

    # 予測
    predicted_label = classifier.predict(test_features)[0]

    # 結果ページに予測結果を渡して表示
    return render_template('result.html', predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)