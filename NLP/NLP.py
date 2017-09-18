import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting = 3)

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',data['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
Y = data.iloc[:, 1:2].values

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state = 0)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,Y_train)

y_pred = nb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
