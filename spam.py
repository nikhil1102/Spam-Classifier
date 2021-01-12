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

# message = pd.read_csv('C:/Users/nikhi/Desktop/cps/project/spam/spam.csv', encoding='latin-1')
# message = message.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
# message = message.rename(columns = {'v1':'label','v2':'message'})
# message['length'] = message['message'].apply(len)
# message['label'] = message['label'].map({'ham': 0, 'spam': 1})

def split_into_lemmas(mess):
    mess = mess.lower()
    words = TextBlob(mess).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

# message['message'].head().apply(split_into_lemmas)
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
