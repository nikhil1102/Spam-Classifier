from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
#from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
import joblib
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from spam import text_process
from spam import split_into_lemmas
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    #Alternative Usage of Saved Model
    #joblib.dump(clf, 'NB_spam_model.pkl')
    NB_spam_model = open('spam.pkl','rb')
    clf = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        # vect = cv.transform(data).toarray()
        my_prediction = clf.predict(data)
        print(my_prediction)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
    #pp.run(host='0.0.0.0', port=8080)