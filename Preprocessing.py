import numpy as np
import nltk

paragraph = """Thank you all so very much. Thank you to the Academy. 
               Thank you to all of you in this room. I have to congratulate 
               the other incredible nominees this year. The Revenant was 
               the product of the tireless efforts of an unbelievable cast
               and crew. First off, to my brother in this endeavor, Mr. Tom 
               Hardy. Tom, your talent on screen can only be surpassed by 
               your friendship off screen … thank you for creating a t
               ranscendent cinematic experience. Thank you to everybody at 
               Fox and New Regency … my entire team. I have to thank 
               everyone from the very onset of my career … To my parents; 
               none of this would be possible without you. And to my 
               friends, I love you dearly; you know who you are. And lastly,
               I just want to say this: Making The Revenant was about
               man's relationship to the natural world. A world that we
               collectively felt in 2015 as the hottest year in recorded
               history. Our production needed to move to the southern
               tip of this planet just to be able to find snow. Climate
               change is real, it is happening right now. It is the most
               urgent threat facing our entire species, and we need to work
               collectively together and stop procrastinating. We need to
               support leaders around the world who do not speak for the 
               big polluters, but who speak for all of humanity, for the
               indigenous people of the world, for the billions and 
               billions of underprivileged people out there who would be
               most affected by this. For our children’s children, and 
               for those people out there whose voices have been drowned
               out by the politics of greed. I thank you all for this 
               amazing award tonight. Let us not take this planet for 
               granted. I do not take tonight for granted. Thank you so very much."""
               

def stem_tokens(tokens):
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    return [stemmer.stem(item) for item in tokens]

def lemmatize(tokens):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(item) for item in tokens]

def normalize(text):
    import string
    import re
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    text = text.lower().translate(remove_punctuation_map)
    text = re.sub(r'\s+',' ',text)
    tokens = nltk.word_tokenize(text)
    return ' '.join([word for word in stem_tokens(tokens) if word not in stop])
    

# Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)

# Tokenizing words
tokens = nltk.word_tokenize(paragraph)

# POS Tagging
tagged_words = nltk.pos_tag(tokens)

# Tagged word paragraph
word_tags = []
for tw in tagged_words:
    word_tags.append(tw[0]+"_"+tw[1])

tagged_paragraph = ' '.join(word_tags)

# Named entity recognition
namedEnt = nltk.ne_chunk(tagged_words)
namedEnt.draw
