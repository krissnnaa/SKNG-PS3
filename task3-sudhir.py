"""
Created on Sat April 6 01:12:48 2019

@author: sudhirsingh

This code works for multinomial classification.
So, it will work for task-2 and task-3 as well.
Need to modify the code to take care of "None" in class label for task-3.
"""

import sys
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


# from collections import defaultdict
# from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
# wordnet_lemmatizer = WordNetLemmatizer()
CLASS_NAMES = []
CLASS_DICTIONARY = {}


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
    # N - gram Level bag of words
    class_list = []
    class_dict = {}

    features = featureSelection(texts, feature_level)
    feature_matrix = count_matrix(features)

    for label in classes:
        class_list.append(class_dict.setdefault(label, len(class_dict)))

    featureWithLabel = list(zip(feature_matrix, class_list))
    return featureWithLabel, class_dict


def LinearSVMClassifier(X,y,x_test,y_test):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    clf = LinearSVC(random_state=0).fit(X_resampled, y_resampled)
    y_pred = clf.predict(x_test).tolist()
    accuracyScore = clf.score(x_test, y_test)
    print("************************************************************************")
    print('Linear SVC accuracy score for test set=%0.2f' % accuracyScore)
    plot_confusion_matrix(y_test, y_pred, classes=CLASS_NAMES, normalize=True, title='Linear SVC Normalized confusion matrix')
    plt.show()
    print("************************************************************************")


def ensembleClassifier(X,y,x_test,y_test):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    clf=RandomForestClassifier(n_estimators=10, random_state=0).fit(X_resampled, y_resampled)
    y_pred = clf.predict(x_test).tolist()
    accuracyScore = clf.score(x_test, y_test)
    print("************************************************************************")
    print('Ensemble Random Forest accuracy score for test set=%0.2f' % accuracyScore)
    plot_confusion_matrix(y_test, y_pred, classes=CLASS_NAMES, normalize=True, title='Ensemble Random Forest Normalized confusion matrix')
    plt.show()
    print("************************************************************************")


def MultinomialNBClassifier(X,y,x_test,y_test):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    clf=MultinomialNB(alpha=1.0, fit_prior=True).fit(X_resampled, y_resampled)
    y_pred = clf.predict(x_test).tolist()
    accuracyScore = clf.score(x_test, y_test)
    print("************************************************************************")
    print('Multinomial Naive Bayes accuracy score for test set=%0.2f' % accuracyScore)
    plot_confusion_matrix(y_test, y_pred, classes=CLASS_NAMES, normalize=True, title='Multinomial Naive Bayes Normalized confusion matrix')
    plt.show()
    print("************************************************************************")


def LogisticRegressionClassifier(X,y,x_test,y_test):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    clf=LogisticRegression(solver='lbfgs', multi_class='auto').fit(X_resampled, y_resampled)
    y_pred = clf.predict(x_test).tolist()
    accuracyScore = clf.score(x_test, y_test)
    print("************************************************************************")
    print('Logistic Regression accuracy score for test set=%0.2f' % accuracyScore)
    plot_confusion_matrix(y_test, y_pred, classes=CLASS_NAMES, normalize=True, title='Logistic Regression Normalized confusion matrix')
    plt.show()
    print("************************************************************************")


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    The below source code to draw confusion matrix is taken from below link,
    and modified as needed.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    temp = unique_labels(y_true, y_pred)
    classes = []
    for value in temp:
        classes.append(list(CLASS_DICTIONARY.keys())[list(CLASS_DICTIONARY.values()).index(value)])
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def perform_operations(file_name):
    # file = 'data/PS3_training_data.txt'
    data = open(file_name).read()
    texts, classes = [], []

    for idx, line in enumerate(data.split("\n")):
        line_contents = line.split('\t')
        if any(line_contents):
            # task-3. escape 'none' label class rows.
            if line_contents[3].lower() == 'none':
                continue
            # text
            texts.append(line_contents[1])
            # reason
            classes.append(line_contents[3])

    del data
    global CLASS_NAMES
    global CLASS_DICTIONARY

    labelFeature, CLASS_DICTIONARY = featureExtraction(texts, classes, 'all')
    X = [l[0] for l in labelFeature]
    y = [l[1] for l in labelFeature]
    CLASS_NAMES = np.asarray(y, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Linear SVC
    LinearSVMClassifier(X_train, y_train, X_test, y_test)
    # Ensemble Random forest
    ensembleClassifier(X_train, y_train, X_test, y_test)
    # Multinomial Naive Bayes
    MultinomialNBClassifier(X_train, y_train, X_test, y_test)
    # Logistic Regression
    LogisticRegressionClassifier(X_train, y_train, X_test, y_test)


def main():
    if len(sys.argv) == 2:
        file_name = sys.argv[1]
    else:
        file_name = input("Enter file name: ")

    if file_name != '':
        perform_operations(file_name)
    else:
        print("message file name empty")


if __name__=='__main__':
    main()



