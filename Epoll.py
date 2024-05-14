import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sentence_transformers
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
df = pd.read_csv('data/electionfinal.csv',nrows=10)
# print(df.shape)
# print(df.info())
# print(df.columns)
# print(df.dtypes)
# print(df.head(5))
# print(df['class'].value_counts())
#
project_path='data/'
# print(project_path)
#
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)
df = shuffle(df).reset_index(drop=True)
split = int(df.shape[0] * 0.8)
#
df_train = df.iloc[:split, :].reset_index(drop=True)
df_test = df.iloc[split:, :].reset_index(drop=True)
# print(df_train.shape, df_test.shape)

bert_model = sentence_transformers.SentenceTransformer('bert-base-nli-mean-tokens')
#
train_comment_embeddings = bert_model.encode(df_train['text'])
# print("Shape of train_comment_embeddings:", train_comment_embeddings.shape)
# print("Contents of train_comment_embeddings:", train_comment_embeddings)
np.save(project_path+'xtrr3.npy',train_comment_embeddings)
#
test_comment_embeddings = bert_model.encode(df_test['text'])
# print("Shape of train_comment_embeddings:", test_comment_embeddings.shape)
# print("Contents of train_comment_embeddings:", test_comment_embeddings)
np.save(project_path+'/xtee3.npy',test_comment_embeddings)
#
np.save(project_path+'/ytrr3.npy',np.array(df_train['label_encoded']))
np.save(project_path+'/ytee3.npy',np.array(df_test['label_encoded']))
#
#
#
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    plot_confusion_matrix(yte,ypred)
    plt.show()
    plt.savefig(project_path+'books_read.pdf')


#
xtrain1=np.load(project_path+'/xtrr3.npy')
xtest1=np.load(project_path+'/xtee3.npy')
ytest1=np.load(project_path+'/ytee3.npy',allow_pickle=True)
ytrain1=np.load(project_path+'/ytrr3.npy',allow_pickle=True)

#
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(xtrain1, ytrain1)
#
#
import pickle
y_pred = svclassifier.predict(xtest1)
pickle.dump(svclassifier, open(project_path+'electiommodel', 'wb'))
a=ytest1.shape[0]
b=(ytest1!=y_pred).sum()
#
print("Accuracy ="+format((a-b)/a*100,'2f')+"%")
print(classification_report(ytest1, y_pred))
plot_cmat(ytest1, y_pred)

rf = RandomForestClassifier(n_estimators=40)
rf.fit(xtrain1, ytrain1)
y_pred =rf.predict(xtest1)
a=ytest1.shape[0]
b=(ytest1!=y_pred).sum()
print("Accuracy ="+format((a-b)/a*100,'2f')+"%")
print(classification_report(ytest1, y_pred))
plot_cmat(ytest1, y_pred)

dt=DecisionTreeClassifier()
dt.fit(xtrain1, ytrain1)
y_pred =dt.predict(xtest1)
a=ytest1.shape[0]
b=(ytest1!=y_pred).sum()
print("Accuracy ="+format((a-b)/a*100,'2f')+"%")
print(classification_report(ytest1, y_pred))
plot_cmat(ytest1, y_pred)