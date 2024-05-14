import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
#import sentence_transformers
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.svm import SVC

project_path='data/'

def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte,ypred)
    plt.show()
    plt.savefig('data/'+'fig2.pdf')

xtrain1 = np.load(project_path + '/xtrr3.npy')
xtest1 = np.load(project_path + '/xtee3.npy')
ytest1 = np.load(project_path + '/ytee3.npy', allow_pickle=True)
ytrain1 = np.load(project_path + '/ytrr3.npy', allow_pickle=True)

print(xtrain1.shape)


rf = RandomForestClassifier(n_estimators=40)
rf.fit(xtrain1, ytrain1)
y_pred =rf.predict(xtest1)
a=ytest1.shape[0]
b=(ytest1!=y_pred).sum()
print("Accuracy ="+format((a-b)/a*100,'2f')+"%")
#print(classification_report((ytest1, y_pred)))
plot_cmat(ytest1, y_pred)
print(y_pred.shape)

dt=DecisionTreeClassifier()
dt.fit(xtrain1, ytrain1)
y_pred =dt.predict(xtest1)
a=ytest1.shape[0]
b=(ytest1!=y_pred).sum()
print("Accuracy ="+format((a-b)/a*100,'2f')+"%")
#print(classification_report((ytest1,y_pred)))
plot_cmat(ytest1, y_pred)
