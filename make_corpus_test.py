from nltk.corpus import PlaintextCorpusReader #maak corpus
corpus_root= 'data'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
corpus = wordlists.fileids()
print(corpus)
print(wordlists.words('arabian_nights_burton_vol01.txt'))
print(wordlists.sents('arabian_nights_burton_vol01.txt')[0:20])




#print(new_corpus)
#wordlists.words('file_name')--> of .sents om aparte woorden of zinnen per file te krijgen


#from nltk.corpus import stopwords #clean corpus
#from nltk.stem.wordnet import WordNetLemmatizer
#import string

#import gensim
#from gensim import corpora
