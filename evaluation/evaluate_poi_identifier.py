#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
# print(clf.score(features_test, labels_test))

print('POIs in test set:', sum(labels_test))
print('people in test set:', len(labels_test))

predictions = clf.predict(features_test)
true_pos = len([i for i in range(len(predictions)) \
    if predictions[i] == 1 and labels_test[i] == 1])
print('true positives:', true_pos)

from sklearn import metrics

print("precision:", metrics.precision_score(labels_test, predictions))
print("recall:", metrics.recall_score(labels_test, predictions))