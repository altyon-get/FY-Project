import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.svm import SVC

random_state = 42
df = pd.read_csv('data/electionfinal.csv',nrows=1000)
project_path='data/'
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)
df = shuffle(df).reset_index(drop=True)
split = int(df.shape[0] * 0.8)
df_train = df.iloc[:split, :].reset_index(drop=True)
df_test = df.iloc[split:, :].reset_index(drop=True)

import sentence_transformers
bert_model = sentence_transformers.SentenceTransformer('bert-base-nli-mean-tokens')
train_comment_embeddings = bert_model.encode(df_train['text'])
np.save(project_path+'xtrr3.npy',train_comment_embeddings)
test_comment_embeddings = bert_model.encode(df_test['text'])
np.save(project_path+'/xtee3.npy',test_comment_embeddings)
np.save(project_path+'/ytrr3.npy',np.array(df_train['label_encoded']))
np.save(project_path+'/ytee3.npy',np.array(df_test['label_encoded']))


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def plot_cmat(y_true, y_pred):
    '''Plotting confusion matrix'''
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

xtrain1=np.load(project_path+'/xtrr3.npy')
xtest1=np.load(project_path+'/xtee3.npy')
ytest1=np.load(project_path+'/ytee3.npy',allow_pickle=True)
ytrain1=np.load(project_path+'/ytrr3.npy',allow_pickle=True)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(xtrain1, ytrain1)

import pickle
# y_pred = svclassifier.predict(xtest1)
# pickle.dump(svclassifier, open(project_path+'electiommodel', 'wb'))
# a=ytest1.shape[0]
# b=(ytest1!=y_pred).sum()
# print("AccuracyX ="+format((a-b)/a*100,'2f')+"%")
# print(classification_report(ytest1, y_pred))
# plot_cmat(ytest1, y_pred)

rf = RandomForestClassifier(n_estimators=40)
rf.fit(xtrain1, ytrain1)
y_pred =rf.predict(xtest1)
pickle.dump(rf, open(project_path+'electiommodel', 'wb'))
a=ytest1.shape[0]
b=(ytest1!=y_pred).sum()
print("Accuracy ="+format((a-b)/a*100,'2f')+"%")
print(classification_report(ytest1, y_pred))
plot_cmat(ytest1, y_pred)
# #
# dt=DecisionTreeClassifier()
# dt.fit(xtrain1, ytrain1)
# y_pred =dt.predict(xtest1)
# a=ytest1.shape[0]
# b=(ytest1!=y_pred).sum()
# print("Accuracy ="+format((a-b)/a*100,'2f')+"%")
# print(classification_report(ytest1, y_pred))
# plot_cmat(ytest1, y_pred)