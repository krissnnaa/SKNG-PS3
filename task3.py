"""
Created on Sat April 6 01:12:48 2019

@author: sudhirsingh

This is the code for task-3 problem set-3. The code works for multinomial classification.
"""

import sys
from ps3_util import PS3Util
from ps3_classifier import PS3Classifier
from sklearn.model_selection import train_test_split


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
            # text data
            texts.append(line_contents[1])
            # reason-code class
            classes.append(line_contents[3])
    del data
    ps3 = PS3Util(texts=texts, classes=classes, feature_level='all')
    labelFeature = ps3.featureExtraction(texts=texts, classes=classes, feature_level='all')

    X = [l[0] for l in labelFeature]
    y = [l[1] for l in labelFeature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    ps3_classifier = PS3Classifier(X_train, y_train, X_test, y_test, ps3)
    ps3_classifier.LinearSVMClassifier(X_train, y_train, X_test, y_test, ps3)
    ps3_classifier.ensembleClassifier(X_train, y_train, X_test, y_test, ps3)
    ps3_classifier.MultinomialNBClassifier(X_train, y_train, X_test, y_test, ps3)
    ps3_classifier.LogisticRegressionClassifier(X_train, y_train, X_test, y_test, ps3)


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


