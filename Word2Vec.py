import nltk
import urllib
import bs4 as bs
import re
from gensim.models import Word2Vec

# Gettings the data source
source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Global_warming').read()

# Parsing the data/ creating BeautifulSoup object
soup = bs.BeautifulSoup(source,'lxml')

# Fetching the data
text = ""
for paragraph in soup.find_all('p'):
    text += paragraph.text

# Preprocessing the data
text = text.lower()
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = re.sub(r'\.',' ',text)
text = re.sub(r'\-',' ',text)

def normalize(text):
    import string
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    text = text.lower().translate(remove_punctuation_map)
    text = re.sub(r'\s+',' ',text)
    tokens = nltk.word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop])

text = normalize(text)


# Preparing the dataset
words = [nltk.word_tokenize(text)]

# Training the Word2Vec model
model = Word2Vec(words, min_count=1)
'''Here min_count is ignoring all words with less than this value.'''
words = model.wv.vocab

# Finding Word Vectors
vector = model.wv['global']

# Most similar words
similar = model.wv.most_similar('warming')


# Using other models.

from gensim.models import KeyedVectors

filename = 'GoogleNews-vectors-negative300.bin'

model = KeyedVectors.load_word2vec_format(filename, binary=True)
model.wv.most_similar('king')
model.wv.most_similar(positive=['king','woman'], negative= ['man'])

'''Download this from web.'''


