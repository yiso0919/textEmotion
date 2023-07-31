"""
与えられたテキストがpositiveかnegativeのどちらか分類する
"""

from flask import Flask, render_template, request
import nltk
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer as WNL

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def preProcessing(test_text):
    """
    前処理
    """
    # HTMLタグを除去
    test_text = re.sub(re.compile('<.*?>'), ' ', test_text)
    # アルファベットと「'」、「-」以外の文字を除去
    test_text = re.sub(r'[^a-zA-Z\'\-]', ' ', test_text)
    # 大文字を小文字に統一
    test_text = test_text.lower()
    # 単語ごとに分割
    test_text = nltk.word_tokenize(test_text)
    # 見出し語化のためのWNLオブジェクトを生成
    wnl = WNL()
    # 要素を削除した時のズレを補正するためのカウンタ
    count = 0
    for i in range(len(test_text)):
        # 単語ごとに「'」もしくは「-」もしくはストップワードが含まれているかどうか判定
        if '\'' in test_text[i-count] or '-' in test_text[i-count] or test_text[i-count] in stopwords.words('english'):
            test_text.remove(test_text[i-count])
            # リストの要素を削除した時のズレを補正するために増加
            count += 1
        # 除去されない単語の場合
        else:
            # wnlを用いて一般的な形にする
            test_text[i-count] = wnl.lemmatize(test_text[i-count])
    # 分割した単語をまとめる
    test_text = ' '.join(test_text)
    return test_text


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # テキストの取得
        test_text = request.form['text']
        # テキストの前処理
        test_text = preProcessing(test_text)

        with open('svc_model.pkl', 'rb') as file:
            classifier = pickle.load(file)
        with open('svc_vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        # 特徴ベクトル化
        test_features = vectorizer.transform([test_text])

        # 予測
        predicted_label = classifier.predict(test_features)[0]

        # 結果ページに予測結果を渡して表示
        return render_template('result.html', predicted_label=predicted_label)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
