import tweepy
import nltk
import re
import pickle

from tweepy import OAuthHandler

# Please change with your own consumer key, consumer secret, access token and access secret
# Initializing the keys
consumer_key = 'jFguAHd6gdkGjB5dr3FOO9W7u'
consumer_secret = '8rRtH2xMcIbD1rfaKP7DXFGKecgg3927s5OMHHSybY81IZDAuZ' 
access_token = '719228830-ftnlZok5UKnufxSLEFBxUoTAuIic5YcNVtyJ9wHZ'
access_secret ='HhOkGAixDVtqmQ1T5ZSPLMD9C6bNijDrnziNJ1PDZ0EoI'

# Initializing the tokens
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
args = ['trump'];
api = tweepy.API(auth,timeout=10)

# Fetching the tweets
list_tweets = []

query = args[0]
if len(args) == 1:
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='en',result_type='recent',geocode="22.1568,89.4332,500km").items(100):
        list_tweets.append(status.text)
        
# Loading the vectorizer and classfier
with open('classifier.pickle','rb') as f:
    classifier = pickle.load(f)
    
with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)    
    
    
total_pos = 0
total_neg = 0
    
# Preprocessing the tweets

from nltk.corpus import stopwords 
stop = set(stopwords.words('english')) 
    
for i in range(len(list_tweets)):
    list_tweets[i] = list_tweets[i].lower()
    list_tweets[i] = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", list_tweets[i])
    list_tweets[i] = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", list_tweets[i])
    list_tweets[i] = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", list_tweets[i])
    list_tweets[i] = re.sub(r"that's","that is", list_tweets[i])
    list_tweets[i] = re.sub(r"there's","there is", list_tweets[i])
    list_tweets[i] = re.sub(r"what's","what is", list_tweets[i])
    list_tweets[i] = re.sub(r"where's","where is", list_tweets[i])
    list_tweets[i] = re.sub(r"it's","it is", list_tweets[i])
    list_tweets[i] = re.sub(r"who's","who is",list_tweets[i])
    list_tweets[i] = re.sub(r"i'm","i am", list_tweets[i])
    list_tweets[i] = re.sub(r"she's","she is", list_tweets[i])
    list_tweets[i] = re.sub(r"he's","he is", list_tweets[i])
    list_tweets[i] = re.sub(r"they're","they are", list_tweets[i])
    list_tweets[i] = re.sub(r"who're","who are", list_tweets[i])
    list_tweets[i] = re.sub(r"ain't","am not", list_tweets[i])
    list_tweets[i] = re.sub(r"wouldn't","would not", list_tweets[i])
    list_tweets[i] = re.sub(r"shouldn't","should not", list_tweets[i])
    list_tweets[i] = re.sub(r"can't","can not", list_tweets[i])
    list_tweets[i] = re.sub(r"couldn't","could not", list_tweets[i])
    list_tweets[i] = re.sub(r"won't","will not", list_tweets[i])
    list_tweets[i] = re.sub(r"\W"," ", list_tweets[i])
    list_tweets[i] = re.sub(r"\d"," ", list_tweets[i])
    list_tweets[i] = re.sub(r"\s+[a-z]\s+"," ", list_tweets[i])
    list_tweets[i] = re.sub(r"\s+[a-z]$"," ", list_tweets[i])
    list_tweets[i] = re.sub(r"^[a-z]\s+"," ", list_tweets[i])
    list_tweets[i] = re.sub(r"\s+"," ", list_tweets[i])
    list_tweets[i].translate(remove_punctuation_map)
    
    tokens = nltk.word_tokenize(list_tweets[i])    
    list_tweets[i] = ' '.join([word for word in tokens if word not in stop])
    
    sent = classifier.predict(tfidf.transform([list_tweets[i]]).toarray())
    if sent[0] == 1:
        total_pos += 1
    else:
        total_neg += 1
    
# Visualizing the results
import matplotlib.pyplot as plt
import numpy as np
objects = ['Positive','Negative']
y_pos = np.arange(len(objects))

plt.bar(y_pos,[total_pos,total_neg],alpha=0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Number')
plt.title('Number of Postive and Negative Tweets')

plt.show()
    
'''Check this 
def normalize(text):
    import string
    from nltk.corpus import stopwords
    from nltk import TweetTokenizer 
    tweet_tokenizer = TweetTokenizer()
    stop = set(stopwords.words('english'))
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    text = text.lower().translate(remove_punctuation_map)
    text = re.sub(r'\s+',' ',text)
    tokens = tweet_tokenizer.tokenize(text)
    #Deleting links.
    tokens = del(tokens[-1])
    return ' '.join([word for word in tokens if word not in stop])
'''


