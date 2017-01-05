#<<<<<<< HEAD
#from nltk.corpus import PlaintextCorpusReader #maak corpus
#corpus_root= 'data'
#wordlists = PlaintextCorpusReader(corpus_root, '.*')
#new_corpus = wordlists.fileids()
#=======
from nltk.corpus import PlaintextCorpusReader #maak corpus
corpus_root= 'data'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
corpus = wordlists.fileids()
#print(corpus)
#print(wordlists.words('arabian_nights_burton_vol01.txt'))
#print(wordlists.sents('arabian_nights_burton_vol01.txt')[0:20])


#######################################
#Prepare the texts for topic modelling
#######################################
# First, we make a new corpus consisting of the nights and some additional texts, because we need enough data to apply topic modelling.
import re
from os import listdir
pattern = re.compile(r'[Tt]he') #Lorien, hier zal dus nog wat achter komen, afhankelijk van de namen van die andere sprookjes.
corpus_tales = []
for file in listdir('data'):
	if pattern.search(file):
		corpus_tales.append('data' + '/' + file)
print(len(corpus_tales)) #990 files are in the corpus (voorlopig)

#Now that we have our corpus, we need to tokenize every file so we can leave out the punctuation, stopwords and save them in 'clean_doc'.
import string
punc = string.punctuation #import a list of punctuation
from nltk.corpus import stopwords #import a list of stopwords
stoplist = stopwords.words('english')
additional_punc = ['``','--', "''"] #this is punctuation that might be in some tales, but is not in the punctuation list of nltk
import nltk #import nltk to be able to use the tokenizer

for tale in corpus_tales:
	filtered_text = []
	f = open(tale,'rt', encoding='utf-8')
	text = f.read()
	f.close()
	text = text.lower()
	text = nltk.word_tokenize(text)
	for item in text:
		if item in punc:
			continue
		if item in additional_punc:
			continue	
		if item in stoplist:
			continue
		filtered_text.append(item)	
	filename = 'clean_doc/' + str(tale[5:-4]) + '_filtered' + '.txt' # 5:-4 so 'data/' is left out as well as '.txt'.
	f_out = open(filename,'wt', encoding='utf-8')
	f_out.write(' '.join(filtered_text))
	f_out.close()



import gensim
from gensim import corpora, models, similarities

from gensim import corpora
texts = filtered_text #hebben we al gedaan met cleanen, dus ik denk dat wij meteen ‘tekst =’ kunnen doen
print(texts)


dictionary = corpora.Dictionary(texts)
dictionary.save('cleanfiles.txtdic')#dictionary een gepaste naam geven



'''from six import iteritems #six is a compatibility library and iteritems returns an iterator over dictionary‘s items
#collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open(''))
 #remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)'''
  