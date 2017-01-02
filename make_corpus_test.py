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




#>>>>>>> origin/master
#print(new_corpus)
#wordlists.words('file_name')--> of .sents om aparte woorden of zinnen per file te krijgen


#from nltk.corpus import stopwords #clean corpus
#from nltk.stem.wordnet import WordNetLemmatizer
#import string

#import gensim
#from gensim import corpora
#<<<<<<< HEAD


from os import listdir

def list_10_volumes(directory):
	volumes = []
	for volume in listdir(directory):
		if volume.startswith('arabian'):
			volumes.append(directory + '/' + volume)
	return volumes

corpus = list_10_volumes('data')
#print(corpus)
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
	#print('Together, the ten volumes have ' + str(counter) + ' characters.' )

read_corpus = []
for volume in corpus:
	f = open(volume, 'rt', encoding='utf-8') 
	text = f.read()
	f.close()
	read_corpus.append(text)	

import re

# Two definitions that can find the starting and ending index of each night. 
def start_idex_nights(regex, text, flags=re.IGNORECASE): # So it is case insensitive.
	start_index = []
	for match in re.finditer(regex, text, flags):
		start_index.append(match.start())
	return start_index	

def start_and_end_index_nights(volume):
	start_index = start_idex_nights("When it was the.*Night,?\n", volume)
	night_indexes = []
	for i in range(len(start_index)-1): # Minus 1 because the last night of each volume is no new start index of the next night (which is in the next volume).
		night_indexes.append((start_index[i], start_index[i+1])) # You get the index of a night and the subsequent night, which is the end index of the previous night.	
	night_indexes.append((start_index[-1], len(volume) - 1)) # Here the indexes of the last night of a volume are added.
	return night_indexes

counter = 0
for volume in read_corpus:
	counter+= (len(start_and_end_index_nights(volume)))
#print(counter) # It seems there are only 990 nights in the ten volumes, or at least, 990 nights are extracted. 

#Now we put each night into a separate file using the indexes calculated above.

#for volume in read_corpus:
	#indexes_night = start_and_end_index_nights(volume)
	#for i in indexes_night:
		#sentence = volume[i[0]+15:i[0]+150] #150 is just a random number, it makes sure that the first sentence (starting from the number e.g. second) is included in 'sentence'
		#sentence = sentence.split(',')	
		#filename = 'data/'+ str("".join(sentence[0])) + '.txt'
		#f = open(filename,'wt', encoding='utf-8')
		#f.write(volume[i[0]:i[1]])

#There are a few files that have a name like ' Eight Hundred and Thirty-sixth Night \n\n She said', because 'she said' stood befor the comma
#I tried to solve it by doing this, but it didn't work. I don't know why though...

#import os
#import re
#pattern = re.compile(r'\n*[Ss]he said')
#for fileName in os.listdir('data'):
#	if pattern.search(fileName):
#		os.rename(fileName, pattern.sub('', fileName))

#####################################
#Calculate statistics for each night
#####################################   	

#Make a new corpus, consisting of the nights so statistics can be calculated
#Each time, I will make a dictionary so it is easy to look up how many words/ lines/ characters... a night has

#I now manually changed the names of those files who were named wrong in order to create a corpus

pattern = re.compile(r'[Tt]he\s')
def corpus(directory): #I slightly changed the function used to make a corpus of the ten volumes
	nights = []
	for night in listdir(directory):
		if pattern.search(night):
			nights.append(directory + '/' + night)
	return nights

corpus_nights = corpus('data')
print(len(corpus_nights)) #to check whether all the nights are in the corpus. That is indeed the case.
#print(corpus_nights)

#How many characters does each night have? 
import collections
def calculate_characters_nights(corpus):
	characters_per_night = {}
	characters_per_night_list = []
	for night in corpus:
		f = open(night, 'rt', encoding='utf-8') 
		text = f.read()
		f.close()
		characters_per_night[night] = len(text)
		characters_per_night_list.append((night,len(text)))
		characters_per_night = collections.OrderedDict(characters_per_night) #we make sure that the order of the data stays the same
	return characters_per_night, characters_per_night_list

char_dict_night = (calculate_characters_nights(corpus_nights))[0]
char_list_night = (calculate_characters_nights(corpus_nights))[1]

#How many lines does each night have?
def calculate_lines_night(file):
	count = 0
	f = open (file, 'rt', encoding='utf-8') 
	for line in f:
		count += 1
	f.close()
	return count

line_list_nights = []
line_dict_nights = {}
for night in corpus_nights:
	line_list_nights.append((night, calculate_lines_night(night)))
	line_dict_nights[night] = calculate_lines_night(night)
	line_dict_nights = collections.OrderedDict(line_dict_nights) #we make sure that the order of the data stays the same

#to calculate the amount of words and sentences, I again made a corpus using the PlaintextCorpusReader
corpus_root= 'data'
corpus_nightsII = PlaintextCorpusReader(corpus_root, '[Tt]he\s.*')	
#print(len(corpus_nightsII.fileids())) #to check whether all 990 nights are in the corpus

word_dic_nights = {}
word_list_nights = []
for file in corpus_nightsII.fileids(): #calculate the amount of words in each volume
	word_list_nights.append((file, len(corpus_nightsII.words(file))))
	word_dic_nights[file] = len(corpus_nightsII.words(file))
	word_dic_nights = collections.OrderedDict(word_dic_nights) #we make sure that the order of the data stays the same

sentence_list_nights = []
sentence_dic_nights = {}
for file in corpus_nightsII.fileids(): #calculate the amount of sentences in each volume
	sentence_list_nights.append((file, len(corpus_nightsII.sents(file))))
	sentence_dic_nights[file] = len(corpus_nightsII.sents(file))
	sentence_dic_nights = collections.OrderedDict(sentence_dic_nights) #we make sure that the order of the data stays the same'''

#####################################
#visualise statistics for each night
##################################### 

#collect data for table with total numbers for each night

characters_per_night = [] #list of the characters per night
for tuples in char_list_night:
	characters_per_night.append(tuples[1])	
#print(characters_per_night)

lines_per_night = [] #list of the lines per night
for tuples in line_list_nights:
	lines_per_night.append(tuples[1])
#print(lines_per_night)

sentences_per_night = [] #list of the sentences per night
for tuples in sentence_list_nights:
	sentences_per_night.append(tuples[1])
#print(sentences_per_night)

words_per_night = [] #list of the words per night
for tuples in word_list_nights:
	words_per_night.append(tuples[1])
#print(words_per_night)

number_nights = [] #create a list with the names of the nights, this is in the order from the dictionaries so not chronological
for name in corpus_nightsII.fileids():
	number_nights.append(name[:-4])#remove the extension from the file name
#print(number_nights) #Here we discovered that there is a fault in a file. The first line of the story is 'The Hundred and and night', so we actually don't know which number it is

#create table with all the data
import numpy as np
import pandas as pd #import panda so we can turn the data into a data table with pandas dataframe
column1 = number_nights
column2 = characters_per_night
column3 = lines_per_night
column4 = sentences_per_night
column5 = words_per_night

df = pd.DataFrame({'Nights': column1,'Total characters': column2,'Total lines': column3, 'Total sentences': column4,'Total words': column5})
print(df)

#print(df.to_csv('datatable_allnights.csv')) #turn data frame table into cvs-file, I get a file but it is messed up

#from prettytable import PrettyTable #not sure yet of this is the best way to visualize the dataframe, i will think about this
#x = PrettyTable()
#def format_for_print(df):    
    #table = PrettyTable([''] + list(df.columns))
    #for row in df.itertuples():
        #table.add_row(row)
    #return str(table)
    #print(format_for_print(df))

#######################################
#Prepare the texts for topic modelling
#######################################
# First, we make a new corpus consisting of the nights and some additional texts, because we need enough data to apply topic modelling.

pattern = re.compile(r'[Tt]he\s') #Lorien, hier zal dus nog wat achter komen, afhankelijk van de namen van die andere sprookjes.
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








  