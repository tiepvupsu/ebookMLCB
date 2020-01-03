
import numpy as np 
d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

d5 = np.array([[2, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])
train_data = np.array([d1, d2, d3, d4])
label = np.array([0, 0, 0, 1])

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(train_data, label)

print(clf.predict(d5))

print(clf.predict_proba(d6))