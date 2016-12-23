##############################
#Text analyses of 1001-nights#
##############################

#Make a corpus of the ten volumes:
from os import listdir

def list_10_volumes(directory):
	volumes = []
	for volume in listdir(directory):
		if volume.startswith('arabian'):
			volumes.append(directory + '/' + volume)
	return volumes

corpus = list_10_volumes('data')
print(corpus)

#How many characters does each volume in the corpus have? 
#and how many characters does the entire corpus have?
def calculate_characters(corpus):
	counter = 0
	for volume in corpus:
		f = open(volume, 'rt', encoding='utf-8') 
		text = f.read()
		f.close()
		print(volume + ' has ' + str(len(text)) + ' characters.')	
		counter+= len(text)
	print('Together, the ten volumes have ' + str(counter) + ' characters.' )

print(calculate_characters(corpus))		

#How many lines does the entire corpus have?
def calculate_lines(corpus):
	count = 0
	for volume in corpus:
		f = open(volume, 'rt', encoding='utf-8') 
		for line in f:
			count += 1
		f.close()	
	print('The corpus of ten volumes has ' + str(count) + ' lines.')	
			
print(calculate_lines(corpus))	

#How many lines does each volume have?
def calculate_lines_II(file):
	count = 0
	f = open (file, 'rt', encoding='utf-8') 
	for line in f:
		count += 1
	f.close()
	return file + ' has ' + str(count) + ' lines.'

for volume in corpus: 
	print(calculate_lines_II(volume))		

#In order to calculate the amount of words and sentences in each volume,
#I made a new corpus of the volumes using the nltk PlaintextCorpusReader
#which has some easy tools that can split a text into a list of words or sentences

from nltk.corpus import PlaintextCorpusReader
corpus_root= 'data'
volumes = PlaintextCorpusReader(corpus_root, 'arabian.*')

list_of_sentences = volumes.sents()
print('The ten volumes consist of ' + str(len(list_of_sentences)) + ' sentences')

list_of_words = volumes.words()
print('The ten volumes consist of ' + str(len(list_of_words)) + ' words')

for item in volumes.fileids(): #calculate the amount of words in each volume
	print(item,':', len(volumes.words(item)), 'words')

for item in volumes.fileids(): #calculate the amount of sentences in each volume
	print(item,':', len(volumes.sents(item)), 'sentences')	

####################################################################################
#Now that we know some of the statistics about each volume and the entire corpus, we
#want to have a look at the individual nights. 
