"""
Created on Tue April 9 04:49:01 2019

@author: sudhirsingh

A common class for feature selection, extract and other code.
This code works for multinomial classification.
"""

import nltk
import string
import array
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from operator import itemgetter
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
# from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
# wordnet_lemmatizer = WordNetLemmatizer()


class PS3Util:

    def __init__(self, class_dictionary=None, texts=None, classes=None, feature_level='unigram', feature_name=None,
                 feature_count=None, feature_dict=None, index_point=None):
        if class_dictionary is None:
            class_dictionary = {}
        if feature_name is None:
            feature_name = []
        if feature_count is None:
            feature_count = array.array(str('i'))
        if feature_dict is None:
            feature_dict = {}
        if index_point is None:
            index_point = []

        self.CLASS_DICTIONARY = class_dictionary
        self.texts = texts
        self.classes = classes
        self.feature_level = feature_level
        self.feature_name = feature_name
        self.feature_count = feature_count
        self.feature_dict = feature_dict
        self.index_point = index_point

    def featureSelection(self, texts, feature_level='unigram'):
        all_feature = []
        for line in texts:
            features = line
            features = "".join([word.lower() for word in features])
            features = features.replace(".", "")
            features = features.translate(str.maketrans('', '', string.punctuation))
            features = features.replace('"', '')
            features = features.replace('\n', '')
            features = features.strip()

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

    def count_matrix(self, features):
        self.index_point.append(0)

        for feature_line in features:
            feature_counter = {}
            for feature in feature_line:
                feature_index = self.feature_dict.setdefault(feature, len(self.feature_dict))
                if feature_index in feature_counter:
                    feature_counter[feature_index] += 1
                else:
                    feature_counter[feature_index] = 1
            self.feature_name.extend(feature_counter.keys())
            self.feature_count.extend(feature_counter.values())
            self.index_point.append(len(self.feature_name))

        matrix = sp.csr_matrix((self.feature_count, self.feature_name, self.index_point),
                               shape=(len(self.index_point) - 1, len(self.feature_dict)),
                               dtype=np.int64)
        print([t for t, i in sorted(self.feature_dict.items(), key=itemgetter(1))])
        print(matrix)
        matrix = matrix.toarray()
        return matrix

    def featureExtraction(self, texts, classes, feature_level='unigram'):
        # N - gram Level Bag of words
        features = self.featureSelection(texts, feature_level)
        feature_matrix = self.count_matrix(features)
        class_list = []
        for label in classes:
            class_list.append(self.CLASS_DICTIONARY.setdefault(label, len(self.CLASS_DICTIONARY)))
        featureWithLabel = list(zip(feature_matrix, class_list))
        return featureWithLabel

    def plot_confusion_matrix(self, y_true, y_pred, classes=None,
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
            classes.append(list(self.CLASS_DICTIONARY.keys())[list(self.CLASS_DICTIONARY.values()).index(value)])
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