import nltk
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

def featureExtractionTask3():
    with open('PS3_training_data.txt','r') as fd:
        data=[l.strip().split('\t') for l in fd.readlines()]

    featureWithLabel=[]
    for item in data:
        innerFeature=[]
        if item[-2]=='OUTDOOR_ACTIVITY' or item[-2]=='OOUTDOOR_ACTIVITY'or item[-2]==' OUTDOOR_ACTIVITY':
            label=0
        elif item[-2]== 'PERSONAL_CARE' or item[-2]=='PERSONA_CARE' :
            label=1
        elif item[-2]== ' MONEY_ISSUE'or item[-2]=='MONEY_ISSUE' :
            label=2
        elif item[-2]== ' (FEAR_OF)_PHYSICAL_PAIN' or item[-2]=='(FEAR_OF)_PHYSICAL_PAIN':
            label=3
        elif item[-2]== 'GOING_TO_PLACES' :
            label=4
        elif item[-2]== 'ATTENDING_EVENT' :
            label=5
        elif item[-2]== 'LEGAL_ISSUE' :
            label=6
        elif item[-2]== 'COMMUNICATION_ISSUE' or item[-2]== 'COMMUNICATION_ISSUE ':
            label=7
        else:
            label=8

        if label != 8:
            tokens=nltk.tokenize.word_tokenize(item[1].lower())
            posTag=nltk.pos_tag(item[1].lower())

            health=0
            legal=0
            fear=0
            drive=0
            tour=0
            money=0
            lie=0
            hospital=0
            concert=0
            attack=0
            buy=0
            doctor=0
            play=0

            for word in tokens:
                if word.startswith('health'):
                    health=1
                if word=='legal':
                    legal=1
                if word=='fear':
                    fear=1
                if word=='drive':
                    drive=1
                if word=='tour':
                    tour=1
                if word=='money':
                    money=1
                if word=='lie' or word=='lied' or word=='lying':
                    lie=1
                if word=='attack':
                    attack=1
                if word=='concert':
                    concert=1
                if word=='money':
                    money=1
                if word=='buy':
                    buy=1
                wordIndex = tokens.index(word)
                if word == 'hospital' and posTag[wordIndex - 2] == 'TO':
                    hospital = 1
                if word == 'doctor' and posTag[wordIndex - 2] == 'TO':
                    doctor = 1

                if word == 'on' and tokens[wordIndex - 1] == 'playing':
                    play = 1
            innerFeature.append(health)
            innerFeature.append(legal)
            innerFeature.append(fear)
            innerFeature.append(drive)
            innerFeature.append(tour)
            innerFeature.append(money)
            innerFeature.append(lie)
            innerFeature.append(attack)
            innerFeature.append(concert)
            innerFeature.append(buy)
            innerFeature.append(hospital)
            innerFeature.append(doctor)
            innerFeature.append(play)
            tupleFeature=[innerFeature,label]
            featureWithLabel.append(tupleFeature)
    return featureWithLabel

if __name__=='__main__':
    labelFeature=featureExtractionTask3()
    x_train = [l[0] for l in labelFeature]
    y_train = [l[1] for l in labelFeature]
    # Linear SVC
    LinearSVMClassifier(x_train[:1000], y_train[:1000], x_train[1000:], y_train[1000:])
    # Ensemble Random forest
    ensembleClassifier(x_train[:1000], y_train[:1000], x_train[1000:], y_train[1000:])
