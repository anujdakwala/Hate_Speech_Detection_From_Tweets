import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
from nltk.stem.porter import *
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore", category=DeprecationWarning)




train = pd.read_csv("C:\\Users\\Anuj\Desktop\\TMBS\\train_E6oV3lV.csv")
test = pd.read_csv('C:\\Users\\Anuj\\Desktop\\TMBS\\test_tweets_anuFYb8.csv')

train.head()

test.head()


####Removing Twitter Handles 
combination = train.append(test, ignore_index=True)


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt    

combination['Preprocess_Tweets'] = np.vectorize(remove_pattern)(combination['tweet'], "@[\w]*")


#### Removing Punctuations, Numbers, and Special Characters


combination['Preprocess_Tweets'] = combination['Preprocess_Tweets'].str.replace("[^a-zA-Z#]", " ")
           
######Removing Short Words

combination['Preprocess_Tweets'] = combination['Preprocess_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))  



combination.head()



#####Tokenization

tokenized_tweet = combination['Preprocess_Tweets'].apply(lambda x: x.split())
tokenized_tweet.head()



#####Stemming

stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()



for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combination['Preprocess_Tweets'] = tokenized_tweet


######Words in NOT racist/sexist tweets


normal_words =' '.join([text for text in combination['Preprocess_Tweets'][combination['label'] == 0]])

wordcloud = WordCloud(width=1000, height=500, random_state=15, max_font_size=90 ,background_color="white").generate(normal_words)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
wordcloud.to_file("C:\\Users\\Anuj\\Desktop\\TMBS\\normal_words.png")


######Words in racist/sexist tweets

negative_words = ' '.join([text for text in combination['Preprocess_Tweets'][combination['label'] == 1]])
wordcloud = WordCloud(width=1000, height=500,random_state=15, max_font_size=90,background_color="white").generate(negative_words)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
wordcloud.to_file("C:\\Users\\Anuj\\Desktop\\TMBS\\negative_words.png")


#####the impact of Hashtags on tweets sentiment


def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags



# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(combination['Preprocess_Tweets'][combination['label'] == 0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combination['Preprocess_Tweets'][combination['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])



a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 15 most frequent hashtags     
d = d.nlargest(columns="Count", n = 15) 
xx= plt.figure(figsize=(10,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count",palette="Blues_d")
ax.set(ylabel = 'Count')
plt.show()
xx.savefig("C:\\Users\\Anuj\\Desktop\\TMBS\\hashtag_normal.png")




b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 15 most frequent hashtags
e = e.nlargest(columns="Count", n = 15)   
yy = plt.figure(figsize=(10,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count" ,palette="Blues_d")
ax.set(ylabel = 'Count')
plt.show()
yy.savefig("C:\\Users\\Anuj\\Desktop\\TMBS\\hashtag_negative.png")

#########Extracting Features from Cleaned Tweets

bagofw_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bagofw = bagofw_vectorizer.fit_transform(combination['Preprocess_Tweets'])





tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combination['Preprocess_Tweets'])



train_bagofw = bagofw[:31962,:]
test_bagofw = bagofw[31962:,:]

# splitting data into training and validation set
xtrain_bagofw, xvalid_bow, ytrain, yvalid = train_test_split(train_bagofw, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bagofw, ytrain) # train the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int) # calculating f1 score




test_pred = lreg.predict_proba(test_bagofw)  ###predicting on the validation set
test_pred_int = test_pred[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('C:\\Users\\Anuj\\Desktop\\TMBS\\Twitter_Hate_Speech_submission_Project2.csv', index=False) # writing data to a CSV file