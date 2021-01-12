import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import sklearn.externals
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report,confusion_matrix, f1_score
import joblib
from sklearn.model_selection import GridSearchCV
#%matplotlib inline

message = pd.read_csv('C:/Users/nikhi/Desktop/cps/project/spam/spam.csv', encoding='latin-1')
message = message.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
message = message.rename(columns = {'v1':'label','v2':'message'})
message['length'] = message['message'].apply(len)
message['label'] = message['label'].map({'ham': 0, 'spam': 1})

def split_into_lemmas(mess):
    mess = mess.lower()
    words = TextBlob(mess).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

message['message'].head().apply(split_into_lemmas)
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
pipeline = Pipeline([
   ( 'bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])
X_train, X_test, y_train, y_test = train_test_split(message['message'], message['label'],test_size=0.33, random_state=42)
pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))
# from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
# svc = svm.SVC()
pipeline_svm = Pipeline([
    ('bow',CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]


grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    verbose=3,
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(n_splits=5),  # what type of cross validation to use
)
svm_detector = grid_svm.fit(X_train, y_train)
predictions2 = svm_detector.predict(X_test)
print(classification_report(predictions2,y_test))
print(confusion_matrix(predictions2,y_test))
print(f1_score(predictions2, y_test))


joblib.dump(grid_svm.best_estimator_, 'spam.pkl')