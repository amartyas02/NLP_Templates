from nltk.corpus import wordnet

# Initializing the list of synnonyms and antonyms
synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for s in syn.lemmas():
        synonyms.append(s.name())
        for a in s.antonyms():
            antonyms.append(a.name())
            
            
# Displaying the synonyms and antonyms
synonyms = list(set(synonyms))
antonyms = list(set(antonyms))