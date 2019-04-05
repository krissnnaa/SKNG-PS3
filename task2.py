import nltk
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

def featureExtractionTask2():

    with open('PS3_training_data.txt','r') as fd:
        data=[l.strip().split('\t') for l in fd.readlines()]

    featureWithLabel=[]

    for item in data:
        innerFeature=[]
        if item[-3]=='POSITIVE':
            label=0
        elif item[-3]=='NEGATIVE':
            label=1
        else:
            label=2
        tokens=nltk.tokenize.word_tokenize(item[2].lower())
        health=0
        product=0
        review=0
        if tokens[0] == 'i' or tokens[0] == 'im':
            im = 1
        else:
            im = 0

        for word in tokens:
            if word.startswith('health'):
                health=1
            if word=='product':
                product=1
            if word.startswith('review'):
                review=1
        innerFeature.append(im)
        innerFeature.append( health)
        innerFeature.append(product)
        innerFeature.append( review)
        tupleFeature=[innerFeature,label]
        featureWithLabel.append(tupleFeature)
    return featureWithLabel
    
def LinearSVMClassifier(X,y,x_test,y_test):

    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    clf=LinearSVC(random_state=0).fit(X_resampled, y_resampled)
    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Linear SVC accuracy score for test set=%0.2f' % accuracyScore)

def ensembleClassifier(X,y,x_test,y_test):

    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    clf=RandomForestClassifier(n_estimators=10,random_state=0).fit(X_resampled, y_resampled)
    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Ensemble Random Forest  accuracy score for test set=%0.2f' % accuracyScore)


if __name__=='__main__':

    labelFeature=featureExtractionTask2()
    x_train = [l[0] for l in labelFeature]
    y_train = [l[1] for l in labelFeature]
    # Linear SVC
    LinearSVMClassifier(x_train[:2000], y_train[:2000], x_train[2000:], y_train[2000:])
    # Ensemble Random forest
    ensembleClassifier(x_train[:2000], y_train[:2000], x_train[2000:], y_train[2000:])
