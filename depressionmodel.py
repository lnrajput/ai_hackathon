import pickle
import re
from numpy import genfromtxt
import numpy as np
import collections
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('data/mbti_1.csv')
#df=df.head(10)
print(df.head(10))
print("*"*40)
print(df.info())

df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)
print(df.head())

plt.figure(figsize=(15,10))
sns.violinplot(x='type', y='words_per_comment', data=df, inner=None, color='lightgray')
sns.stripplot(x='type', y='words_per_comment', data=df, size=4, jitter=True)
plt.ylabel('Words per comment')
#plt.show()

df['http_per_comment'] = df['posts'].apply(lambda x: x.count('http')/50)
df['music_per_comment'] = df['posts'].apply(lambda x: x.count('music')/50)
df['question_per_comment'] = df['posts'].apply(lambda x: x.count('?')/50)
df['img_per_comment'] = df['posts'].apply(lambda x: x.count('jpg')/50)
df['excl_per_comment'] = df['posts'].apply(lambda x: x.count('!')/50)
df['ellipsis_per_comment'] = df['posts'].apply(lambda x: x.count('...')/50)

plt.figure(figsize=(15,10))
sns.jointplot(x='words_per_comment', y='ellipsis_per_comment', data=df, kind='kde')


X = df.drop(['type','posts'], axis=1).values
y = df['type'].values
#X_test=genfromtxt('data/X_test1.csv',delimiter=',')
#y_test=genfromtxt('data/y_test1.csv',delimiter=',',dtype=str)
print('sam')

np.savetxt("data/X_test1.csv",X,delimiter=",")
np.savetxt("data/y_test1.csv",y,delimiter=",",fmt='%s')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.1, random_state=5)

X_test1=genfromtxt('data/X_test1.csv',delimiter=',')
y_test1=genfromtxt('data/y_test1.csv',delimiter=',',dtype=str)
# Random Forest


random_forest = RandomForestClassifier(bootstrap=False,criterion="gini",max_features=0.5,min_samples_leaf=2,min_samples_split=2,n_estimators=100)
random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.1, random_state=5)

Y_prediction1 = random_forest.predict(X_test1)

print(accuracy_score(y_test1,Y_prediction1))
with open('model/RandomForest.pkl','wb') as handle:
        pickle.dump(random_forest,handle,protocol=pickle.HIGHEST_PROTOCOL)

np.savetxt('data/RandomForest.csv',Y_prediction1,delimiter=",",fmt='%s')
# Logistic Regression
