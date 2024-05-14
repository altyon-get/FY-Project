import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mpld3 import display

df=pd.read_csv('data/IndianElection19TwitterData.csv',index_col=0)

# Keywords mentioning Modi and Rahul respectively
NaMo_ref = ["Modi","PM","modi", "#PMModi","modi ji", "narendra modi", "@narendramodi","#Vote4Modi"]
RaGa_ref = ["rahul", "Rahul","RahulGandhi", "gandhi","@RahulGandhi","Gandhi","#Vote4Rahul","#Vote4Gandhi","#Vote4RahulGandhi"]

# method to refer whether contains perticular words in tweets
def identify_subject(tweet, refs):
    flag = 0
    for ref in refs:
        if tweet.find(ref) != -1:
            flag = 1
    return flag

"""
 finding whether the tweet referred about Modi or Rahul
    """;
df['NaModi'] = df['Tweet'].apply(lambda x: identify_subject(x, NaMo_ref))
df['RaGandhi'] = df['Tweet'].apply(lambda x: identify_subject(x, RaGa_ref))
df.head(10)

"""
 Filtering tweets mentioning either one of the pm candidate or both
 assigning 1 for NaModi and RaGandhi variables where there is a mention of them in the tweets 
 which was already decide by looking the keywords in tweets
    """;
df=df[(df['NaModi']==1) | (df['RaGandhi']==1)]


# Preprocessing
df=df.reset_index()
df.drop('index',axis=1,inplace=True)


# Import stopwords
import nltk
from nltk.corpus import stopwords

# Import textblob
from textblob import Word, TextBlob
import nltk

# Downloading imp libraries and packages
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')

"""Processing tweets by removing stopwords from nltk library
""";
def preprocess_tweets(tweet):
    processed_tweet = tweet
    processed_tweet.replace('[^\w\s]', '')
    processed_tweet = " ".join(word for word in processed_tweet.split() if word not in stop_words)
    processed_tweet = " ".join(Word(word).lemmatize() for word in processed_tweet.split())
    return(processed_tweet)

df['Processed Tweet'] = df['Tweet'].apply(lambda x: preprocess_tweets(x))

print('Base Tweet\n', df['Tweet'][0])
print('\n------------------------------------\n')
print('Cleaned Tweet\n', df['Processed Tweet'][0])

# Calculate polarity and subjectivity of the tweet
df['polarity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[0])
df['subjectivity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[1])
df[['Processed Tweet', 'NaModi', 'RaGandhi', 'polarity', 'subjectivity']].head()

# display(df[df['RaGandhi']==1][['RaGandhi','polarity','subjectivity']].groupby('RaGandhi').agg([np.mean, np.max, np.min, np.median]))
# df[df['NaModi']==1][['NaModi','polarity','subjectivity']].groupby('NaModi').agg([np.mean, np.max, np.min, np.median])

# Print the DataFrame instead of using display()
print(df[df['RaGandhi']==1][['RaGandhi','polarity','subjectivity']].groupby('RaGandhi').agg([np.mean, np.max, np.min, np.median]))
print(df[df['NaModi']==1][['NaModi','polarity','subjectivity']].groupby('NaModi').agg([np.mean, np.max, np.min, np.median]))


naModi = df[df['NaModi']==1][['Date', 'polarity']]
naModi = naModi.sort_values(by='Date', ascending=True)
naModi['MA Polarity'] = naModi.polarity.rolling(10, min_periods=3).mean()

raGandhi = df[df['RaGandhi']==1][['Date', 'polarity']]
raGandhi = raGandhi.sort_values(by='Date', ascending=True)
raGandhi['MA Polarity'] = raGandhi.polarity.rolling(10, min_periods=3).mean()


fig, axes = plt.subplots(2, 1, figsize=(13, 10))

axes[0].plot(naModi['Date'], naModi['MA Polarity'])
axes[0].set_title("\n".join(["Polarity change of Tweets mentioning Narendra Modi"]))
axes[1].plot(raGandhi['Date'], raGandhi['MA Polarity'], color='red')
axes[1].set_title("\n".join(["Polarity change of Tweets mentioning Rahul Gandhi"]))

fig.suptitle("\n".join(["Analysis of Tweets Polarity about PM Candidates"]), y=0.98)
plt.savefig('polarity_analysis1.png')

fig, axes = plt.subplots(2, 1, figsize=(15, 10))

axes[0].plot(naModi['Date'][:1000], naModi['MA Polarity'][:1000])
axes[0].set_title("\n".join(["Polarity change of Tweets mentioning Narendra Modi"]))
axes[1].plot(raGandhi['Date'][:1000], raGandhi['MA Polarity'][:1000], color='red')
axes[1].set_title("\n".join(["Polarity change of Tweets mentioning Rahul Gandhi"]))
fig.suptitle("\n".join(["Analysis of Tweets Polarity about PM Candidates"]), y=0.98)
plt.savefig('polarity_analysis2.png')