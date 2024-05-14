# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#for bert
"""
    We are using pretrained 'bert-base-multilingual-uncased-sentiment' model 
    for predicting the sentiment of the review as a number of stars (between 1 and 5)
    """;
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# Vader sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # for polarity score
analyser = SentimentIntensityAnalyzer()
"""following functions returns positive, negative, neutral emotion score of the text respectively. 
""";
def pos(tweet):
    SentDict = analyser.polarity_scores(tweet)
    return SentDict['pos']
def neg(tweet):
    SentDict = analyser.polarity_scores(tweet)
    return SentDict['neg']
def neu(tweet):
    SentDict = analyser.polarity_scores(tweet)
    return SentDict['neu']
df=pd.read_csv('data/IndianElection19TwitterData.csv',index_col=0)
# print(df.head(5));

# Tweets related to Narendra Modi
"""Filtering out tweets with some keywords and hashtags in it
   referring to Narendra Modi that are commonly used on twitter
""";
modi = ["Modi","PM","modi", "#PMModi","modi ji", "narendra modi", "@narendramodi","#Vote4Modi"]
modi_df = pd.DataFrame(columns=["Date", "User","Tweet"])
def ismodi(tweet):
    t = tweet.split()
    for i in modi:
        if i in t:
            return True
modi_rows = []
# Here df is the main data
for row in df.values:
    if ismodi(str(row[2])):
         modi_rows.append({"Date":row[0], "User":row[1],"Tweet":row[2]})
modi_df = pd.concat([modi_df, pd.DataFrame(modi_rows)])
# print(modi_df.head(10))
modi_df['Tweet'].nunique()


# Tweets related to Rahul Gandhi
"""
 Filtering out tweets with some keywords and hashtags in it 
 referring to Rahul Gandhi that are commonly used on twitter
""";
rahul = ["rahul", "Rahul","RahulGandhi", "gandhi","@RahulGandhi","Gandhi","#Vote4Rahul","#Vote4Gandhi","#Vote4RahulGandhi"]
rahul_df = pd.DataFrame(columns=["Date", "User","Tweet"])
def israhul(tweet):
    t = tweet.split()
    for i in rahul:
        if i in t:
            return True
rahul_rows = []
for row in df.values:
    if israhul(str(row[2])):
         rahul_rows.append({"Date":row[0], "User":row[1],"Tweet":row[2]})
rahul_df = pd.concat([rahul_df, pd.DataFrame(rahul_rows)])
# print(rahul_df.head(10))


# Data Cleaning : Removing Stopwords & Panctuations
from sklearn.feature_extraction import text
import string
stop = text.ENGLISH_STOP_WORDS
"""Removing stopwords (as in sklearn library) from tweets so as to get good polarity scores
""";
modi_df['Tweet'] = modi_df['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
rahul_df['Tweet'] = rahul_df['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
"""Removing panctuations from tweets
""";
modi_df['Tweet'] = modi_df['Tweet'].apply(remove_punctuations)
rahul_df['Tweet'] = rahul_df['Tweet'].apply(remove_punctuations)
# print(rahul_df['Tweet'].head(10))

# VadarSentiment Sentiment Analysis
"""Calculating the polarity scores with help of code snippets mentioned at the importing libraries section
""";
modi_df['pos'] = modi_df['Tweet'].apply(lambda x :pos(x))
modi_df['neg'] = modi_df['Tweet'].apply(lambda x :neg(x))
modi_df['neu'] = modi_df['Tweet'].apply(lambda x :neu(x))
emotion=[]
for i in range(0,25683):
    emotion.append(max(modi_df['pos'][i],modi_df['neu'][i],modi_df['neg'][i]))
modi_df['FinalEmotion']=emotion
"""Traversing through the polarity scores for each tweet and
assigning the Final Emotion as per the highest score among positive, negative, neutral
""";
for i in range(0,25683):
    if modi_df['FinalEmotion'][i]==modi_df['pos'][i]:
        modi_df['FinalEmotion'][i]='positive'
    elif modi_df['FinalEmotion'][i]==modi_df['neg'][i]:
        modi_df['FinalEmotion'][i]='negative'
    elif modi_df['FinalEmotion'][i]==modi_df['neu'][i]:
        modi_df['FinalEmotion'][i]='neutral'
# print(modi_df.head(20));
print(modi_df['FinalEmotion'].value_counts());

# Plot visualizing the counts of emotions of all the tweets
plt.figure(figsize=(15,10))
sns.set_style("darkgrid")
ax = sns.countplot(x='FinalEmotion', data=modi_df, palette=['#36454F', '#89CFF0'])
ax.set_title('Sentiment scores of Tweets about Modi')
# Save the plot as a PNG file
plt.savefig('sentiment_plot1.png', dpi=300)

#TODO: RAHUL PART


# Flair Sentiment Analysis
# from flair.models import TextClassifier
# from flair.data import Sentence
# sia = TextClassifier.load('en-sentiment')
# """Flair text classifier model code snippet to get the emotion of tweet
# """;
# def flair_prediction(x):
#     sentence = Sentence(x)
#     sia.predict(sentence)
#     score = sentence.labels[0]
#     if "POSITIVE" in str(score):
#         return "pos"
#     elif "NEGATIVE" in str(score):
#         return "neg"
#     else:
#         return "neu"
#Just to clear the previous sentiments by vaderSentiment, we need to drop that columns for using Flair on it
# rahul_df.drop(['pos', 'neg', 'neu', 'FinalEmotion'],axis=1,inplace=True)
# modi_df.drop(['pos', 'neg', 'neu', 'FinalEmotion'],axis=1,inplace=True)

# Applying flair on both the dataframes
# rahul_df['Emotion']=rahul_df['Tweet'].apply(flair_prediction)
# modi_df['Emotion']=modi_df['Tweet'].apply(flair_prediction)

# Sentiments for Narendra Modi
# plt.figure(figsize=(15,10))
# sns.set_style("darkgrid")
# ax = sns.countplot(x=modi_df['Emotion'],palette=['#36454F','#89CFF0'])
# ax.set_title('Sentiments scores of Tweets about Modi')
# plt.savefig('sentiment_plot2.png', dpi=300)

from textblob import TextBlob

def textblob_prediction(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "pos"
    elif polarity < 0:
        return "neg"
    else:
        return "neu"

#Just to clear the previous sentiments by vaderSentiment, we need to drop that columns for using Flair on it
# rahul_df.drop(['pos', 'neg', 'neu', 'FinalEmotion'],axis=1,inplace=True)
# modi_df.drop(['pos', 'neg', 'neu', 'FinalEmotion'],axis=1,inplace=True)

# Apply sentiment analysis using TextBlob
rahul_df['Emotion'] = rahul_df['Tweet'].apply(textblob_prediction)
modi_df['Emotion'] = modi_df['Tweet'].apply(textblob_prediction)

# Sentiments for Narendra Modi
plt.figure(figsize=(15,10))
sns.set_style("darkgrid")
ax = sns.countplot(x=modi_df['Emotion'],palette=['#36454F','#89CFF0'])
ax.set_title('Sentiments scores of Tweets about Modi')
plt.savefig('sentiment_plot2.png', dpi=300)

# Sentiments for Narendra Modi
plt.figure(figsize=(15,10))
sns.set_style("darkgrid")
ax = sns.countplot(x=rahul_df['Emotion'],palette=['#36454F','#89CFF0'])
ax.set_title('Sentiments scores of Tweets about Rahul')
plt.savefig('sentiment_plot3.png', dpi=300)