import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#clean string
def clean_string(string):
    string = string.strip()
    string = re.sub(r'[^a-zA-Z0-9]+',' ', string)    #eliminate special char
    string = string.lower()         #lowercase the string
    string = re.sub(r'\s[a-z]\s', '', string) #eliminate single letter
    return string


#Vectorization and nomorlization of token counts
def tokenize(train_comments,test_comments):
    vectorizer = CountVectorizer(lowercase=True, stop_words='english', min_df=5, max_df=0.05, binary=True, max_features=1000)
    X_train = vectorizer.fit_transform(train_comments)
    X_test = vectorizer.transform(test_comments)
    return X_train,X_test


#Vectorization and normorlization of TF-IDF
def TF_IDF(train_comments,test_comments):
    tf_idf_vectorizer = TfidfVectorizer()
    X_triain= tf_idf_vectorizer.fit_transform(train_comments)
    X_test = tf_idf_vectorizer.transform(test_comments)
    return X_triain,X_test

#build training set
def TrainSetBuilding(train_comments):
    X_comments = []
    for i in range(len(train_comments)):
        train_comment = clean_string(train_comments[i])
        X_comments.append(train_comment)
    return X_comments



#Naive Bayes model
class NaiveBayes:

    def fit(self, X_train, y_train):
        '''
        :param X_train: shape(n,m), tokenized
        :param y_train: shape(n,), list of classes, class name shoud be number
        :return: theta matrix and dictionary of prior probability
        '''
        counter = Counter(y_train)
        num_class = len(counter.keys())     #class:number of instance in this class
        num_sample, num_feature = X_train.shape[0], X_train.shape[1]

        prior = {k: v/num_sample for k, v in counter.items()}   #prior probability

        theta = np.zeros((num_feature, num_class))          #theta matrix

        X_train_array = X_train.toarray()
        for j in range(X_train_array.shape[1]):
            for i in range(X_train_array.shape[0]):
                if X_train_array[i][j]!= 0:
                    key = y_train[i]
                    theta[j, key] += 1
        for m in range(num_feature):
            for key in range(num_class):
                theta[m][key] = (theta[m][key]+1)/ (counter[key]+2)
        return theta, prior

    def predict(self, X_test, theta, prior):

        od = OrderedDict(sorted(prior.items()))
        prior_in_order = np.asarray(list(od.values()))
        log_theta = np.log(theta)
        log_1_minus_theta = np.log(1 - theta)
        X_test_array = X_test.toarray()
        prob_matrix = np.log(prior_in_order) + (X_test_array @ log_theta + (np.ones(X_test_array.shape) - X_test_array) @ log_1_minus_theta)
        y_predict = np.argmax(prob_matrix, axis=1)
        return y_predict


    
if __name__ == '__main__':
    # load data
    data_train = pd.read_csv("reddit_train.csv", index_col=None)
    train_comments = data_train['comments']
    y = data_train['subreddits']

    number = LabelEncoder()
    y = number.fit_transform(y)

    # build training set
    X_comments = TrainSetBuilding(train_comments)
    X_comments_array = np.asarray(X_comments)

    # cross-validation
    kf = KFold(n_splits=5, shuffle=True)
    accuracy = []
    for train_index, test_index in kf.split(X_comments):
        X_train, X_test = X_comments_array[train_index], X_comments_array[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_vec, X_test_vec = tokenize(X_train, X_test)

        BNB = NaiveBayes()
        theta, prior = BNB.fit(X_train_vec, y_train)
        y_predict = BNB.predict(X_test_vec, theta, prior)
        ac = accuracy_score(y_test, y_predict)
        accuracy.append(ac)
        print(ac)
    print('Average accuracy is:', np.mean(accuracy))