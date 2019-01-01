from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk

dataset = ["The amount of polution is increasing day by day",
           "The concert was just great",
           "I love to see Gordon Ramsay cook",
           "Google is introducing a new technology",
           "AI Robots are examples of great technology present today",
           "All of us were singing in the concert",
           "We have launch campaigns to stop pollution and global warming"]

dataset = [line.lower() for line in dataset]

# Creating Tfidf Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)

# Visualizing the Tfidf Model
'''
#print(X[0])
X[0]-X[len(dataset)]
In X[0], (sentence_no., pos. of word in whole document) and the corresponding tf-idf value
'''

# Creating the SVD
'''Here n_components is the no.of concepts we want to group the sentences into.'''
lsa = TruncatedSVD(n_components = 4, n_iter = 100)
lsa.fit(X)

# First Column of V
'''
lsa.components_ consist of 4 lists with each list of no. of words elements.
Here for row1, we have different values of all words corresponding to 1st concept.
Words of this concept have high value.
'''
row1 = lsa.components_[0]


# Word Concept Dictionary Creation
concept_words = {}

# Visualizing the concepts
'''Get all the words'''
terms = vectorizer.get_feature_names()

'''Creating a dictionary of concept_no. and list of tuples of highest value.'''
for i,comp in enumerate(lsa.components_):
    
    componentTerms = zip(terms,comp)
    '''Creating a tuple of words and their values in each concept.'''
    sortedTerms = sorted(componentTerms,key=lambda x:x[1],reverse=True)
    sortedTerms = sortedTerms[:10]
    '''Taking top 10 words and corresponding values.'''
    concept_words["Concept "+str(i)] = sortedTerms
    

# Sentence Concepts
concept_sentence_score=[]
i=0
for key in concept_words.keys():
    '''Iterating through concepts.'''
    sentence_scores = []
    for sentence in dataset:
        words = nltk.word_tokenize(sentence)
        '''words contains all the words of a single sentence.'''
        score = 0
        for word in words:
            '''Taking one word at a time.'''
            for word_with_score in concept_words[key]:
                if word == word_with_score[0]:
                    score += word_with_score[1]
            '''Added the total score of each word in single concept.'''
        sentence_scores.append(score)
        '''Appending the total score of each word in a single sentence.'''
    concept_sentence_score.append((key,sentence_scores))
    i+=1
    '''
    This gives us the score of each sentence for each concept.
    The one sentence with the highest value in each concept belongs to that concept.
    '''
    
'''
We can know the concept by checking the concept_words.
'''