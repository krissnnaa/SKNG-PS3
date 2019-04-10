# SKNG-PS3 (Sudhir Krishna Nicole and George)

Problem Set 3 - README.md
==========================
Date: 04-09-2019
Authors: Sudhir Singh, Nicole Thurgood, Geroge Bowie and Krishna P Neupane


Introduction: Analysis of CLIPEval Dataset and UCI Sentiment Dataset for three different tasks.


Requirements: Following modules are required in python 3 environment:

		1) nltk
		2) numpy
		3) sklearn
		4) scipy
		5) imblearn
		6) matplotlib


Requirements files:
The requirements.txt file contains all dependant libraries to be installed.
To install all dependant libraries, please run the following command.

pip install -r requirements.txt


Command Line Execution:
1) To run task1:
python3 task1.py PS3_training_data.txt

2) To run task2:
python3 task2.py PS3_training_data.txt

3) To run task3:
python3 task3.py PS3_training_data.txt


Output:

1) Model Accuracy: With four classifier (Linear SVC, Ensemble Classifier, Multinomial Naive Bayes classifier and logistic regression classifier)
2) Confusion matrix - with normalization and without normalization
3) Classification report
4) Confusion matrix plot - with normalization
