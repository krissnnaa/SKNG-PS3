'''
task-1 modified code by Sudhir.
This code works for multinomial classification.
So, it's working for task-2 as well, but has lower accuracy, around 70%.
Need to modify the code to take care of "None" in class label for task-3
'''

import nltk
import string
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import array
import numpy as np
import scipy.sparse as sp


# from collections import defaultdict
# from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
# wordnet_lemmatizer = WordNetLemmatizer()


def featureSelection(texts, feature_level='unigram'):
    all_feature = []
    for line in texts:
        features = line
        features = "".join([word.lower() for word in features])
        features = features.replace(".", "")
        features = features.translate(str.maketrans('', '', string.punctuation))

        if feature_level.lower() == 'unigram':
            features = [word for word in nltk.tokenize.word_tokenize(features)]
            features = [word for word in features if not word in stop_words]

        if feature_level.lower() == 'bigram':
            features = [' '.join(bigram) for bigram in nltk.bigrams(features.split())]

        if feature_level.lower() == 'all':
            features = [word for word in nltk.tokenize.word_tokenize(features)]
            features = [word for word in features if not word in stop_words]
            bigram_temp = [' '.join(bigram) for bigram in nltk.bigrams(line.split())]
            features.extend(bigram_temp)

        all_feature.append(features)
        # next pos tag.

        # word embedding or word vectors

    return all_feature


def count_matrix(features):

    feature_name = []
    feature_count = array.array(str('i'))
    feature_dict = {}
    index_point = []
    index_point.append(0)

    for feature_line in features:
        feature_counter = {}
        for feature in feature_line:
            feature_index = feature_dict.setdefault(feature, len(feature_dict))
            if feature_index in feature_counter:
                feature_counter[feature_index] += 1
            else:
                feature_counter[feature_index] = 1
        feature_name.extend(feature_counter.keys())
        feature_count.extend(feature_counter.values())
        index_point.append(len(feature_name))

    return sp.csr_matrix((feature_count, feature_name, index_point), shape=(len(index_point) - 1, len(feature_dict)), dtype=np.int64).toarray()


def featureExtraction(texts, classes, feature_level='unigram'):
    # N - gram Level count
    class_list = []
    class_dict = {}

    features = featureSelection(texts, feature_level)
    feature_matrix = count_matrix(features)

    for label in classes:
        class_list.append(class_dict.setdefault(label, len(class_dict)))

    featureWithLabel = list(zip(feature_matrix, class_list))
    return featureWithLabel


def LinearSVMClassifier(X,y,x_test,y_test):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    clf = LinearSVC(random_state=0).fit(X_resampled, y_resampled)
    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Linear SVC accuracy score for test set=%0.2f' % accuracyScore)


def ensembleClassifier(X,y,x_test,y_test):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    clf=RandomForestClassifier(n_estimators=10, random_state=0).fit(X_resampled, y_resampled)
    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Ensemble Random Forest  accuracy score for test set=%0.2f' % accuracyScore)


def MultinomialNBClassifier(X,y,x_test,y_test):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    clf=MultinomialNB(alpha=1.0, fit_prior=True).fit(X_resampled, y_resampled)
    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Multinomial Naive Bayes accuracy score for test set=%0.2f' % accuracyScore)


def LogisticRegressionClassifier(X,y,x_test,y_test):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    clf=LogisticRegression(solver='lbfgs', multi_class='auto').fit(X_resampled, y_resampled)
    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Logistic Regression accuracy score for test set=%0.2f' % accuracyScore)


def main():

    file = 'data/PS3_training_data.txt'
    data = open(file).read()
    texts, classes = [], []

    for idx, line in enumerate(data.split("\n")):
        line_contents = line.split('\t')
        if any(line_contents):
            # text = line_contents[1]
            texts.append(line_contents[1])
            # GENRE = line_contents[4]
            classes.append(line_contents[4])
            # sentiment = line_contents[2]
            # classes.append(line_contents[2])
            # reason = line_contents[3]
            # classes.append(line_contents[3])

    del data
    labelFeature = featureExtraction(texts, classes, 'All')
    x_train = [l[0] for l in labelFeature]
    y_train = [l[1] for l in labelFeature]
    # Linear SVC
    LinearSVMClassifier(x_train[:2000], y_train[:2000], x_train[2000:], y_train[2000:])
    # Ensemble Random forest
    ensembleClassifier(x_train[:2000], y_train[:2000], x_train[2000:], y_train[2000:])
    # Multinomial Naive Bayes
    MultinomialNBClassifier(x_train[:2000], y_train[:2000], x_train[2000:], y_train[2000:])
    # Logistic Regression
    LogisticRegressionClassifier(x_train[:2000], y_train[:2000], x_train[2000:], y_train[2000:])


if __name__=='__main__':
    main()



