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
def calculate_characters(corpus):
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

##################################################################
#visualisation of the statistiscs with basic plotting techniques
##################################################################

'''import matplotlib.pyplot as plt #pyhton library for plotting data
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
#print(table_statistics)'''

##################################
#Separate the volumes into nights
##################################

# Now that we know some of the statistics about each volume and the entire corpus, we
# want to have a look at the individual nights. 

# To make sure all the volumes are automatically read. 
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
	sentence_dic_nights = collections.OrderedDict(sentence_dic_nights) #we make sure that the order of the data stays the same

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








  