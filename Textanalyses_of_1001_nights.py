##############################
#Text analyses of 1001-nights#
##############################

# Make a corpus of the ten volumes:
from os import listdir

def list_10_volumes(directory):
	volumes = []
	for volume in listdir(directory):
		if volume.startswith('arabian'):
			volumes.append(directory + '/' + volume)
	return volumes

corpus = list_10_volumes('data')
print(corpus)

# How many characters does each volume in the corpus have? 
# and how many characters does the entire corpus have?
'''def calculate_characters(corpus):
	counter = 0
	characters_per_volume = []
	for volume in corpus:
		f = open(volume, 'rt', encoding='utf-8') 
		text = f.read()
		f.close()
		characters_per_volume.append(len(text))	
		counter+= len(text)
	return characters_per_volume, 'Together, the ten volumes have ' + str(counter) + ' characters.'

print(calculate_characters(corpus))	

# How many lines does the entire corpus have?
def calculate_lines(corpus):
	count = 0
	for volume in corpus:
		f = open(volume, 'rt', encoding='utf-8') 
		for line in f:
			count += 1
		f.close()	
	print('The corpus of ten volumes has ' + str(count) + ' lines.')	
			
print(calculate_lines(corpus))	

# How many lines does each volume have?
def calculate_lines_II(file):
	count = 0
	f = open (file, 'rt', encoding='utf-8') 
	for line in f:
		count += 1
	f.close()
	return [file, ' has ', count, 'lines.']

for volume in corpus: 
	print(calculate_lines_II(volume))	

# In order to calculate the amount of words and sentences in each volume,
# I made a new corpus of the volumes using the nltk PlaintextCorpusReader
# which has some easy tools that can split a text into a list of words or sentences.

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
'''
##################################################################
#visualisation of the statistiscs with basic plotting techniques
##################################################################
'''
import matplotlib.pyplot as plt #pyhton library for plotting data
import numpy as np
#%matplotlib inline #only necessary when you need to use the code in Jupyter Notebook

#visualise the characters per volume
#make a list of the total characters per volume -
characters_per_volume = (calculate_characters(corpus))[0]
x = [1,2,3,4,5,6,7,8,9,10]
y = characters_per_volume
x_labels = ["vol1","vol2","vol3","vol4","vol5","vol6","vol7","vol8","vol9","vol10"]
y_labels = [719626, 713975, 740002, 608981, 805537, 620786, 816219, 728202, 743379, 130715]
width = 0.8 # width of the bars
fig, ax = plt.subplots() #make subplots, so that their will be labels above the bars
rects1 = ax.bar(x, y_labels) #rects stands for rectangles, the shape of the bar chart. 

def autolabel(rects):# attach labels above bars #zoals gevonden op matplotlib site
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom') #The placement of the text is determined by the height function, or the height of the column
        #and the number that is put on top of each column is written by: '%d' %int(height). 
        #So all you need to do is create an array of strings that you want at the top of the columns and iterate through. 
#print(autolabel(rects1))

plt.bar(x,y) #plot a bar chart and attach some information to it
plt.title("total number of characters in a volume")
plt.xlabel("volume")
plt.ylabel("# of characters")
plt.xticks(x,x_labels)
#plt.show()

#visualise the lines per volume
#make a list of the total lines per volume ---> lijst maken van de counters die def(calculate_lines_II(volume)) teruggeeft 
lines_per_volume = []
for volume in corpus: 
	lines_per_volume.append((calculate_lines_II(volume))[2]) 

x = [1,2,3,4,5,6,7,8,9,10]
y = lines_per_volume
x_labels = ["vol1","vol2","vol3","vol4","vol5","vol6","vol7","vol8","vol9","vol10"]
y_labels = [12498, 12159, 13092, 10925, 14585, 10491, 13786, 13189, 12517, 2221 ]
width = 0.8 # width of the bars
fig, ax = plt.subplots() #make subplots, so that their will be labels above the bars
rects2 = ax.bar(x, y_labels) #rects stands for rectangles, the shape of the bar chart. 

def autolabel(rects):# attach labels above bars as recommanded by the matplotlib site
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom') #The placement of the text is determined by the height function, or the height of the column
        #and the number that is put on top of each column is written by: '%d' %int(height). 
        #So all you need to do is create an array of strings that you want at the top of the columns and iterate through. 
#print(autolabel(rects2))

plt.bar(x,y) #plot a bar chart and attach some information to it
plt.title("total number of lines in a volume")
plt.xlabel("volume")
plt.ylabel("# of lines")
plt.xticks(x,x_labels)
#plt.show()

#visualise the sentences per volume
sentences_per_volume = []
for item in volumes.fileids():
	sentences_per_volume.append(len(volumes.sents(item)))
#print(sentences_per_volume)

x = [1,2,3,4,5,6,7,8,9,10]
y = sentences_per_volume
x_labels = ["vol1","vol2","vol3","vol4","vol5","vol6","vol7","vol8","vol9","vol10"]
y_labels = sentences_per_volume
width = 0.8 # width of the bars
fig, ax = plt.subplots() #make subplots, so that their will be labels above the bars
rects1 = ax.bar(x, y_labels) #rects stands for rectangles, the shape of the bar chart. 

def autolabel(rects):# attach labels above bars as recommanded by the matplotlib site
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom') #The placement of the text is determined by the height function, or the height of the column
        #and the number that is put on top of each column is written by: '%d' %int(height). 
        #So all you need to do is create an array of strings that you want at the top of the columns and iterate through. 
#print(autolabel(rects1))

plt.bar(x,y) #plot a bar chart and attach some information to it
plt.title("total number of sentences in a volume")
plt.xlabel("volume")
plt.ylabel("# of sentences")
plt.xticks(x,x_labels)
#plt.show()


#visualise the words per volume
words_per_volume = []
for item in volumes.fileids():
	words_per_volume.append(len(volumes.words(item)))
#print(words_per_volume)

x = [1,2,3,4,5,6,7,8,9,10]
y = words_per_volume
x_labels = ["vol1","vol2","vol3","vol4","vol5","vol6","vol7","vol8","vol9","vol10"]
y_labels = words_per_volume
width = 0.8 # width of the bars
fig, ax = plt.subplots() #make subplots, so that their will be labels above the bars
rects1 = ax.bar(x, y_labels) #rects stands for rectangles, the shape of the bar chart. 

def autolabel(rects):# attach labels above bars as recommanded by the matplotlib site
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom') #The placement of the text is determined by the height function, or the height of the column
        #and the number that is put on top of each column is written by: '%d' %int(height). 
        #So all you need to do is create an array of strings that you want at the top of the columns and iterate through. 
#print(autolabel(rects1))

plt.bar(x,y) #plot a bar chart and attach some information to it
plt.title("total number of words in a volume")
plt.xlabel("volume")
plt.ylabel("# of words")
plt.xticks(x,x_labels)
#plt.show()

#visualise all the total numbers for the entire corpus (= ten volumes of The Arabian Nights)
import texttable as tt#import texttable module
tab = tt.Texttable() #initialize texttable object

header = ['Corpus', 'Total characters', 'Total lines', 'Total sentences', 'Total words']#To insert a header we create a list with each element containing the title of a column
tab.header(header)# add it to the table using the header() method of the TextTable object.

row = ['The Arabian Nights', '6627422', '115463', '43094','1459332']
tab.add_row(row) #To insert a row into the table, we create a list with the elements of the row and add it to the table

tab.set_cols_width([18,18,18,18,18])#set the width of the table cells
#set the horizontal and vertical alignment of data within table cells
tab.set_cols_align(['c','c', 'c','c', 'c']) #‘c’ for center alignment 

tab.set_deco(tab.HEADER | tab.VLINES)#control drawing of lines between rows and columns and between the header and the first row
#I choose for a line below the header and lines between columns

tab.set_chars(['-','|','+','#'])#list of elements which determine character used for horizontal lines, vertical lines,
#intersection points of these lines and the header line, in that order

table_statistics = tab.draw()#table is returned as a string
#print(table_statistics)
'''
##################################
#Separate the volumes into nights
##################################

# Now that we know some of the statistics about each volume and the entire corpus, we
# want to have a look at the individual nights. 

# To make sure all the volumes are automatically read. 
'''read_corpus = []
for volume in corpus:
	f = open(volume, 'rt', encoding='utf-8') 
	text = f.read()
	f.close()
	read_corpus.append(text)'''

import re

# Two definitions that can find the starting and ending index of each night. 
'''def start_idex_nights(regex, text, flags=re.IGNORECASE): # So it is case insensitive.
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

pattern = re.compile(r'\s[Nn]ight')
for volume in read_corpus:
	indexes_night = start_and_end_index_nights(volume)
	for i in indexes_night:
		sentence = volume[i[0]+12:i[0]+150] #150 is just a random number, it makes sure that the first sentence (starting from the number e.g. 'second') is included in 'sentence'
		sentence = pattern.split(sentence)	
		filename = 'data/'+ str(sentence[0]) + '.txt'
		f = open(filename,'wt', encoding='utf-8')
		f.write(volume[i[0]:i[1]])
		f.close()

#I now created 990 files and each file has a name like 'the second', 'the the Four Hundred and Sixty-third',...
#This is important so that we know which night we are talking about.

#####################################
#Calculate statistics for each night
#####################################   	

#Make a new corpus, consisting of the nights so statistics can be calculated
#Each time, I will make a dictionary so it is easy to look up how many words/ lines/ characters... a night has
#Each time, I will also have to make a list. These will then be used to visualise the statistics. 

corpus_root= 'data'
corpus_nightsII = PlaintextCorpusReader(corpus_root, '[Tt]he\s.*')	
print(len(corpus_nightsII.fileids())) #to check whether all 990 nights are in the corpus

#How many characters does each night have? 
import collections
def calculate_characters_nights(corpus):
	characters_per_night = {}
	characters_per_night_list = []
	for night in corpus:
		f = open('data/' + night, 'rt', encoding='utf-8') 
		text = f.read()
		f.close()
		characters_per_night[night] = len(text)
		characters_per_night_list.append((night,len(text)))
		characters_per_night = collections.OrderedDict(characters_per_night) #we make sure that the order of the data stays the same
	return characters_per_night, characters_per_night_list

char_dict_night = (calculate_characters_nights(corpus_nightsII.fileids()))[0]
char_list_night = (calculate_characters_nights(corpus_nightsII.fileids()))[1]

#Which night has the most characters? 
for file, characters in char_list_night:
    if characters == max(char_dict_night.values()):
        print(file, characters) # the Eight Hundred and Forty-fifth.txt => 51841

#How many lines does each night have?
def calculate_lines_night(file):
	count = 0
	f = open ('data/' + file, 'rt', encoding='utf-8') 
	for line in f:
		count += 1
	f.close()
	return count

line_list_nights = []
line_dict_nights = {}
for night in corpus_nightsII.fileids():
	line_list_nights.append((night, calculate_lines_night(night)))
	line_dict_nights[night] = calculate_lines_night(night)
	line_dict_nights = collections.OrderedDict(line_dict_nights) #we make sure that the order of the data stays the same

#Which night has the most lines? 
for file, characters in line_list_nights:
    if characters == max(line_dict_nights.values()):
        print(file, characters) #the Eight Hundred and Forty-fifth.txt => 864

word_dic_nights = {}
word_list_nights = []
for file in corpus_nightsII.fileids(): #calculate the amount of words in each volume
	word_list_nights.append((file, len(corpus_nightsII.words(file))))
	word_dic_nights[file] = len(corpus_nightsII.words(file))
	word_dic_nights = collections.OrderedDict(word_dic_nights) #we make sure that the order of the data stays the same

#Which night has the most words?
for file, characters in word_list_nights:
	if characters == max(word_dic_nights.values()):
		print(file, characters) #the Eight Hundred and Forty-fifth.txt => 11794

sentence_list_nights = []
sentence_dic_nights = {}
for file in corpus_nightsII.fileids(): #calculate the amount of sentences in each volume
	sentence_list_nights.append((file, len(corpus_nightsII.sents(file))))
	sentence_dic_nights[file] = len(corpus_nightsII.sents(file))
	sentence_dic_nights = collections.OrderedDict(sentence_dic_nights) #we make sure that the order of the data stays the same 

#Which night has the most sentences?
for file, characters in sentence_list_nights:
	if characters == max(sentence_dic_nights.values()):
		print(file, characters)	#the Eight Hundred and Forty-fifth.txt => 399

#In the following block of code, we calculate what the average word length is in each night
from nltk import FreqDist
import nltk
dict_word_length = {}
for file in corpus_nightsII.fileids():
	text = corpus_nightsII.words(file)
	x = [len(words) for words in text]
	fdist = FreqDist(x)
	dict_word_length[file] = fdist.max()
#print(dict_word_length)
'''
#####################################
#visualise statistics for each night
##################################### 

#collect data for table with total numbers for each night

'''characters_per_night = [] #list of the characters per night
for tuples in char_list_night:
	characters_per_night.append(tuples[1])	
print(characters_per_night)

# Now we have this list, we can also calculate the average amount of characters in a file:
avarage_characters = sum(characters_per_night)/len(characters_per_night)
print(round(avarage_characters)) # On average, there are 6603 characters in a file

lines_per_night = [] #list of the lines per night
for tuples in line_list_nights:
	lines_per_night.append(tuples[1])
print(lines_per_night)

avarage_lines = sum(lines_per_night)/len(lines_per_night)
print(round(avarage_lines)) #On average, there are 166 characters in a file

sentences_per_night = [] #list of the sentences per night
for tuples in sentence_list_nights:
	sentences_per_night.append(tuples[1])
#print(sentences_per_night)

avarage_sentences = sum(sentences_per_night)/len(sentences_per_night)
print(round(avarage_sentences)) #On average, there are 43 sentences in a file

words_per_night = [] #list of the words per night
for tuples in word_list_nights:
	words_per_night.append(tuples[1])
#print(words_per_night)

avarage_words = sum(words_per_night)/len(words_per_night)
print(round(avarage_words)) #On average, there are 1454 words in a file.

number_nights = [] #create a list with the names of the nights, this is in the order from the dictionaries so not chronological
for name in corpus_nightsII.fileids():
	number_nights.append(name[:-4])#remove the extension from the file name
print(number_nights) #Here we discovered that there is a fault in a file. The first line of the story is 'The Hundred and and night', so we actually don't know which number it is

#create table with all the data
import numpy as np
import pandas as pd #import panda so we can turn the data into a data table with pandas dataframe
column1 = number_nights
column2 = characters_per_night
column3 = lines_per_night
column4 = sentences_per_night
column5 = words_per_night

df = pd.DataFrame({'Nights': column1,'Total characters': column2,'Total lines': column3, 'Total sentences': column4,'Total words': column5})
#print(df) #show the data frame

from pandas import ExcelWriter as xlwt #import excelwriter module
from xlwt import Workbook
writer = xlwt('table of all nights.xlsx') #create an excel file from the data frame
workbook = writer.book #define the excel workbook
df.to_excel(writer, 'Sheet1') #place the data frame on the first sheet of the excel file
worksheet = writer.sheets['Sheet1'] #define the worksheet
worksheet.set_column('B:F',35) #set the column width for columns B up to F, so we can see all the text in the cells
format = workbook.add_format({'bold': True, 'font_color': 'red'}) #put in bold and red the night with the highest number of char., lines, sents. and words

#find the highest numbers per column (per list)
print(max(characters_per_night))
print(max(lines_per_night))
print(max(sentences_per_night))
print(max(words_per_night))

worksheet.write('C31', '51841' , format) #highlight the night with the highest number of characters
worksheet.write('D31', '864' , format) #highlight the night with the highest number of lines
worksheet.write('E31', '399' , format) #highlight the night with the highest number of sentences
worksheet.write('F31', '11794' , format) #highlight the night with the highest number of words
worksheet.write('B31', 'the Eight Hundred and Forty-fifth', format)#highlight the name of the night with the highest numbers
writer.save() #save and close the excel file

#now the general visualisations of the statistics are finished, we can start preparing the texts for topic modeling '''

#######################################
#Prepare the texts for topic modeling
#######################################
# First, we make a new corpus consisting of the nights and some additional texts, because we need enough data to apply topic modelling.

'''pattern = re.compile(r'[Tt]he') 
corpus_tales = []
for file in listdir('data'):
	if pattern.search(file):
		corpus_tales.append('data' + '/' + file)
print(len(corpus_tales)) #for topic modeling we should have a minimum of 1000 files, now we have 1030 files (990 nights and some additional tales)
'''
#Now that we have our corpus, we need to tokenize every file so we can leave out the punctuation, stopwords, words that occur only once
# modals, cardinal numbers... and save them in 'clean_doc'.
import string
punc = string.punctuation #import a list of punctuation
from nltk.corpus import stopwords #import a list of stopwords
stoplist = stopwords.words('english')
additional_stopwords = ["thou", "thee", 'thy', "'s"] #these are middle English stopwords that are not in the nltk list and the possesive 's'
additional_punc = ['``','--', "''"] #this is punctuation that might be in some tales, but is not in the punctuation list of nltk
import nltk #import nltk to be able to use the tokenizer
from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()
from collections import defaultdict
frequency = defaultdict(int)# make an empty default dict so we can compute the frequency of the words and delete words that only occur once

'''for tale in corpus_tales:
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
		if item in additional_stopwords:
			continue
		filtered_text.append(item)
	pos_text = nltk.pos_tag(filtered_text)
	for tuples in pos_text: #this will remove modals and cardinal numbers
		if tuples[1] == 'MD':
			filtered_text.remove(tuples[0])
		if tuples[1] == 'CD':
			filtered_text.remove(tuples[0])	
	#filtered_text = [p_stemmer.stem(i) for i in filtered_text] #the words are stemmed. We noticed that some words turned into a rather strange stem and we don't know whether that is how it should be.				
	for words in filtered_text:
		frequency[words] += 1
	filtered_text = [words for words in filtered_text if frequency[words] > 1] #now only words that occur more than once are in 'filtered_text'	
	filename = 'clean_doc/' + str(tale[5:-4]) + '_filtered' + '.txt' # 5:-4 so 'data/' is left out as well as '.txt'.
	f_out = open(filename,'wt', encoding='utf-8')
	f_out.write(' '.join(filtered_text))
	f_out.close()
'''

# Now we make a new corpus consisting of the filtered texts
pattern = re.compile(r'[Tt]he') 
clean_corpus= []
for file in listdir('clean_doc'):
	if pattern.search(file):
		clean_corpus.append('clean_doc' + '/' + file)
#print(len(clean_corpus)) #To check whether all 1030 files are in the corpus. It is indeed correct.

# Now we make a nested list and afterwards a dictionary, this is necessary for creating the document matrix
import gensim
from gensim import corpora

nested_list = []
for tale in clean_corpus:
	f = open(tale,'rt', encoding='utf-8')
	text = f.read()
	f.close()
	text = nltk.word_tokenize(text)
	nested_list.append(text)
	dictionary = corpora.Dictionary(nested_list)
#dictionary.save('clean_files_dic.txtdic')
#print(dictionary.token2id)


# We are ready to turn the dictionary into a document-term matrix
# Now we convert the dictionary into a bag of words and call it a vector corpus
vector_corpus = [dictionary.doc2bow(text) for text in nested_list] # this gives us the document-term matrix
#print(vector_corpus [2]) #list of sparse vectors equal to the number of documents. 
#In each document the sparse vector is a series of tuples.The tuples are (term ID, term frequency) pairs.

#######################################
# Ready for topic modeling
#######################################

#import numpy

#numpy.random.seed(1) #setting random seed to get the same results each time.
ldamodel = gensim.models.LdaModel(vector_corpus, num_topics=50, id2word = dictionary, passes=5)
# first parameter: determine how many topics should be generated. Our document set is relatively large, so we’re  asking for 200 topics.
# second parameter: our previous dictionary to map ids to strings
# third parameter: number of laps the model will take through corpus. More passes = more accurate model. 
#But a lot of passes can be slow on a very large corpus.So let's say we do 15 laps.

#ldamodel.save('topicmodel.lda') #We save and load the model for later use instead of having to rebuild it every time
ldamodel = gensim.models.LdaModel.load('topicmodel.lda')

#print(ldamodel.show_topics(num_topics=-1, num_words=4)) #prints the num_words most probable words for all topics to log. topics=-1 to print all topics.
# first parameter defines the number of topics, second parameter the number of words per topic, this is 10 words per topic by default
#print(ldamodel.print_topics(5) #print the most contributing words for ... randomly selected topics

# Now that we have our ldamodel and have an idea about the topics that are in fairy tales, we want to test the model
# on our original corpus of tales: corpus_nightsII (in order to do that, we need to convert in into a BOW representation)

# We are not sure whether we need to clean the testcorpus as well. We did it to be sure.

'''for night in corpus_nightsII.fileids():
	filtered_text = []
	f = open('data/' + night, 'rt', encoding='utf-8') 
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
		if item in additional_stopwords:
			continue
		filtered_text.append(item)
	pos_text = nltk.pos_tag(filtered_text)
	for tuples in pos_text: #this will remove modals and cardinal numbers
		if tuples[1] == 'MD':
			filtered_text.remove(tuples[0])
		if tuples[1] == 'CD':
			filtered_text.remove(tuples[0])	
	#filtered_text = [p_stemmer.stem(i) for i in filtered_text] #the words are stemmed. We noticed that some words turned into a rather strange stem and we don't know whether that is how it should be.				
	for words in filtered_text:
		frequency[words] += 1
	filtered_text = [words for words in filtered_text if frequency[words] > 1] #now only words that occur more than once are in 'filtered_text'	
	filename = 'clean_nights/' + str(night[:-4]) + '_filtered' + '.txt'
	f_out = open(filename,'wt', encoding='utf-8')
	f_out.write(' '.join(filtered_text))
	f_out.close()'''

# The cleaned nights are now in 'clean_nights', now we can convert them to a BOW representation	
pattern = re.compile(r'[Tt]he') 
clean_nights_corpus = []
for file in listdir('clean_nights'):
	if pattern.search(file):
		clean_nights_corpus.append('clean_nights' + '/' + file)

corpus = []
for file in clean_nights_corpus:
	f = open(file,'rt', encoding='utf-8')
	text = f.read()
	f.close()
	text = nltk.word_tokenize(text)
	bow_vector = dictionary.doc2bow(text)
	#lda_vector = ldamodel[bow_vector]# dus Lorien, deze moet je dan aanpassen. Misschien moet je lda_vector ook even hernoemen omdat ik hieronder ook lda_vector gebruik :p gewoon voor de zekerheid
	lda_vector = ldamodel.get_document_topics(bow_vector, minimum_probability=0.0)
	lda_vector = [b for (a, b) in sorted(lda_vector)]
	corpus.append(lda_vector)

# Now we would like to print every document's single most prominent LDA topic in a separate txt file
'''f_out = open('topic_per_night.txt','at', encoding='utf-8')
for file in clean_nights_corpus:
	f = open(file,'rt', encoding='utf-8')
	text = f.read()
	f.close()
	text = nltk.word_tokenize(text)
	bow_vector = dictionary.doc2bow(text)
	lda_vector = ldamodel[bow_vector]
	topic = ldamodel.print_topic(max(lda_vector, key=lambda item: item[1])[0])
	f_out.write('\n' + file + ':\n' + topic + '\n')
f_out.close()	
'''
# Now we make a matrix of the documents & topics:
import numpy as np
X = ldamodel.show_topics(num_topics= 50, num_words=10) #not necessary, was just a test to see if it made any difference
X = np.array(corpus) #should be the matrix containing the nights & the topics'''

#print(X.shape) to check if the shape of our matrix is suitable for hierarchical clustering, it shoudl give the number of files and number of topics
#print(X)

'''#########################
# Evaluate our LDA model
#########################
# We will split each document into two parts, and check that topics of the first half are similar to topics 
# of the second halves. And we check whether two halves of different documents are less similar.
# The halves of the same document should be very similar, the halves of different documents should be a bit 
# less similar, although similarity is expected here as well, since they are all fairy tales.
def intra_inter(model, test_docs, num_pairs=2000):
    part1 = []
    part2 = []
    for file in test_docs:
    	f = open(file, 'rt', encoding='utf-8') 
    	text = f.read()
    	f.close()
    	text = nltk.word_tokenize(text)# split each test document into two halves and compute topics for each half
    	part1.append(ldamodel[dictionary.doc2bow(text[:len(text) / 2])])
    	part2.append(ldamodel[dictionary.doc2bow(text[len(text) / 2 :])])
    
    # print computed similarities (uses cossim)
    print('average cosine similarity between corresponding parts (higher is better):')
    print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)]))

    random_pairs = np.random.randint(0, len(test_docs), size=(num_pairs, 2))
    print('average cosine similarity between 2000 random parts (should be a bit lower):')    
    print(np.mean([gensim.matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs]))

print(intra_inter(ldamodel, clean_nights_corpus)) '''   

###########################################
# Hierarchical clustering with topic model
###########################################
from matplotlib import pyplot as plt   #this should be some start code for the hierarchical clustering of our topics. Not finished yet!
import numpy as np

from scipy.spatial.distance import pdist, squareform
dm = squareform(pdist(X, 'cosine'))

from scipy.cluster.hierarchy import linkage
linkage_object = linkage(dm, method='ward', metric='euclidean')

#from scipy.cluster.hierarchy import dendrogram
#d = dendrogram(linkage_object, labels=labels, orientation='right')
#plt.savefig('tree.pdf')


#creating a linkage matrix
#Z = linkage(X, 'ward') 
'''this is how you generate a linkage matrix. But this gives a ValueError: setting an array element with a sequence. 
This is because only equally shaped arrays can be clustered, we have a difference in lengths between the lists inside the list of lists 
'cosine'is one of the methods that can be used to calculate the distance between newly formed clusters
we use the cosine similarity because it is better for topic clustering # X stands for the matrix.''' 
#print(Z) #Z[i] will tell us which clusters were merged in the i-th iteration/pass

#plotting a hierarchical clustering dendogram

'''#creating word clouds
import os
import wordcloud

MODELS_DIR = 'ldamodel'
final_topics = open(os.path.join(MODELS_DIR, 'topic_per_night.txt', 'rb') 
curr_topic = 0

for line in final_topics:
    line = line.strip()[line.rindex(":") + 2:]
    scores = [float(x.split("*")[0]) for x in line.split(" + ")]
    words = [x.split("*")[1] for x in line.split(" + ")]
    freqs = []
    for word, score in zip(words, scores):
        freqs.append((word, score))
    elements = wordcloud.fit_words(freqs, width=120, height=120)
    wordcloud.draw(elements, "gs_topic_%d.png" % (curr_topic),
                   width=120, height=120)
    curr_topic += 1
final_topics.close()'''



'''tfidf = TfidfModel(vector_corpus)# first build TF-IDF model
 

weights = tfidf(vector_corpus[0])# Get TF-IDF weights
 

weights = [(dictionary[pair[0]], pair[1]) for pair in weights]# Get terms from the dictionary and pair with weights
# Replace term IDs with human consumable strings
#weights = [(counts[dictionary[pair[0]]], pair[1]) for pair in weights]

wc = WordCloud(background_color="white",max_words=2000,width = 1024,height = 720)# Initialize the word cloud


wc.generate_from_frequencies(weights)# Generate the cloud
 

wc.to_file("word_cloud.png")# Save the could to a file'''