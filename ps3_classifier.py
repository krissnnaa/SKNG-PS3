"""
Created on Sat April 6 05:35:09 2019

@author: sudhirsingh

This class contains all classifier used in problem set-3 implementation.
"""
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class PS3Classifier:

    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None, ps3=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.ps3 = ps3

    def LinearSVMClassifier(self, X, y, x_test, y_test, ps3):
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        clf = LinearSVC(random_state=0).fit(X_resampled, y_resampled)
        y_pred = clf.predict(x_test).tolist()
        accuracyScore = clf.score(x_test, y_test)
        print('Linear SVC accuracy score for test set=%0.2f' % accuracyScore)
        ps3.plot_confusion_matrix(y_test, y_pred, normalize=True,
                                  title='Linear SVC Normalized confusion matrix')
        plt.show()
        print("------------------------------------------------------------------------")
        print("Confusion matrix, without normalization")
        print(metrics.confusion_matrix(y_test, y_pred))
        print("------------------------------------------------------------------------")
        print(metrics.classification_report(y_test, y_pred))
        print("------------------------------------------------------------------------")
        print("************************************************************************")

    def ensembleClassifier(self, X, y, x_test, y_test, ps3):
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_resampled, y_resampled)
        y_pred = clf.predict(x_test).tolist()
        accuracyScore = clf.score(x_test, y_test)
        print('Ensemble Random Forest accuracy score for test set=%0.2f' % accuracyScore)
        ps3.plot_confusion_matrix(y_test, y_pred, normalize=True,
                                  title='Ensemble Random Forest Normalized confusion matrix')
        plt.show()
        print("------------------------------------------------------------------------")
        print("Confusion matrix, without normalization")
        print(metrics.confusion_matrix(y_test, y_pred))
        print("------------------------------------------------------------------------")
        print(metrics.classification_report(y_test, y_pred))
        print("------------------------------------------------------------------------")
        print("************************************************************************")

    def MultinomialNBClassifier(self, X, y, x_test, y_test, ps3):
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        clf = MultinomialNB(alpha=1.0, fit_prior=True).fit(X_resampled, y_resampled)
        y_pred = clf.predict(x_test).tolist()
        accuracyScore = clf.score(x_test, y_test)
        print('Multinomial Naive Bayes accuracy score for test set=%0.2f' % accuracyScore)
        ps3.plot_confusion_matrix(y_test, y_pred, normalize=True,
                                  title='Multinomial Naive Bayes Normalized confusion matrix')
        plt.show()
        print("------------------------------------------------------------------------")
        print("Confusion matrix, without normalization")
        print(metrics.confusion_matrix(y_test, y_pred))
        print("------------------------------------------------------------------------")
        print(metrics.classification_report(y_test, y_pred))
        print("------------------------------------------------------------------------")
        print("************************************************************************")

    def LogisticRegressionClassifier(self, X, y, x_test, y_test, ps3):
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        clf = LogisticRegression(solver='lbfgs', multi_class='auto').fit(X_resampled, y_resampled)
        y_pred = clf.predict(x_test).tolist()
        accuracyScore = clf.score(x_test, y_test)
        print('Logistic Regression accuracy score for test set=%0.2f' % accuracyScore)
        ps3.plot_confusion_matrix(y_test, y_pred, normalize=True,
                                  title='Logistic Regression Normalized confusion matrix')
        plt.show()
        print("------------------------------------------------------------------------")
        print("Confusion matrix, without normalization")
        print(metrics.confusion_matrix(y_test, y_pred))
        print("------------------------------------------------------------------------")
        print(metrics.classification_report(y_test, y_pred))
        print("------------------------------------------------------------------------")
        print("************************************************************************")