import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
con = preprocessing.LabelEncoder()


X = np.array(df.drop(['class'], 1))
Z = np.array(df['class'])

## Can split in train and test and check accuracy only on test
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Z, test_size=0.2)

clf = KMeans(n_clusters = 2)
clf.fit(X)

print(Z)

## We are calculating accuracy in this manner as the .score of an unsupervised classifier
## does not measure the same parameter as of a supervised classifier
ans = []
correct = 0
for i in range(len(X)):
    sample = np.array(X[i])
    sample = sample.reshape(-1, len(sample))
    prediction = clf.predict(sample)
    ans.append(prediction[0])
    if prediction[0] == 0 and Z[i] == 2:
        correct += 1
    elif prediction[0] == 1 and Z[i] == 4:
        correct += 1

print(correct/len(X))


# print(clf.score(X_test))