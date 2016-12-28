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
#visualization of the statistiscs with basic plotting techniques
##################################################################

import matplotlib.pyplot as plt #pyhton library for plotting data
import numpy as np
#%matplotlib inline #only necessary when you need to use the code in Jupyter Notebook

#visualize the characters per volume
#make a list of the total characters per volume ---> lijst maken van de counters die def(calculate_characters) teruggeeft
#characters_per_volume= []
#for volume in corpus:
	#characters_per_volume.append(calculate_characters(corpus)) #ik weet niet hoe ik hier een lijst van kan maken
#dit klopt niet, de aparte counters zonder tekst komen niet in een lijst, wel de tekst en de counters
#print(characters_per_volume)

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
print(autolabel(rects1))

plt.bar(x,y) #plot a bar chart and attach some information to it
plt.title("total number of characters in a volume")
plt.xlabel("volume")
plt.ylabel("# of characters")
plt.xticks(x,x_labels)
plt.show()

#visualize the lines per volume
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
print(autolabel(rects2))

plt.bar(x,y) #plot a bar chart and attach some information to it
plt.title("total number of lines in a volume")
plt.xlabel("volume")
plt.ylabel("# of lines")
plt.xticks(x,x_labels)
plt.show()

#visualize the sentences per volume
sentences_per_volume = []
for item in volumes.fileids():
	sentences_per_volume.append(len(volumes.sents(item)))
print(sentences_per_volume)

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
print(autolabel(rects1))

plt.bar(x,y) #plot a bar chart and attach some information to it
plt.title("total number of sentences in a volume")
plt.xlabel("volume")
plt.ylabel("# of sentences")
plt.xticks(x,x_labels)
plt.show()


#visualize the words per volume
words_per_volume = []
for item in volumes.fileids():
	words_per_volume.append(len(volumes.words(item)))
print(words_per_volume)

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
print(autolabel(rects1))

plt.bar(x,y) #plot a bar chart and attach some information to it
plt.title("total number of words in a volume")
plt.xlabel("volume")
plt.ylabel("# of words")
plt.xticks(x,x_labels)
plt.show()

#visualize all the total numbers for the entire corpus (= ten volumes of The Arabian Nights)
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
print(table_statistics)

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
print(counter) # It seems there are only 990 nights in the ten volumes, or at least, 990 nights are extracted. 

#Now we put each night into a separate file using the indexes calculated above.

for volume in read_corpus:
	indexes_night = start_and_end_index_nights(volume)
	for i in indexes_night:
		sentence = volume[i[0]+15:i[0]+150] #150 is just a random number, it makes sure that the first sentence (starting from the number e.g. second) is included in 'sentence'
		sentence = sentence.split(',')	
		filename = 'data/'+ str("".join(sentence[0])) + '.txt'
		f = open(filename,'wt', encoding='utf-8')
		f.write(volume[i[0]:i[1]])

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

def corpus_nights(directory): #I slightly changed the function used to make a corpus of the ten volumes
	nights = []
	for night in listdir(directory):
		if night.endswith('ight.txt'):
			nights.append(directory + '/' + night)
	return nights

corpus_nights = corpus_nights('data')

#How many characters does each night have? 
def calculate_characters_nights(corpus):
	characters_per_night = {}
	for night in corpus:
		f = open(night, 'rt', encoding='utf-8') 
		text = f.read()
		f.close()
		characters_per_night[night] = len(text)
	return characters_per_night

char_dict_night = calculate_characters_nights(corpus_nights)

#How many lines does each night have?
def calculate_lines_night(file):
	count = 0
	f = open (file, 'rt', encoding='utf-8') 
	for line in f:
		count += 1
	f.close()
	return count

line_dict_nights = {}
for night in corpus_nights:
	line_dict_nights[night] = calculate_lines_night(night)
print(line_dict_nights)	

#to calculate the amount of words and sentences, I again made a corpus using the PlaintextCorpusReader
corpus_root= 'data'
corpus_nightsII = PlaintextCorpusReader(corpus_root, '.*[nN]ight.txt')	
print(len(corpus_nightsII.fileids())) 

word_dic_nights = {}
for file in corpus_nightsII.fileids(): #calculate the amount of words in each volume
	word_dic_nights[file] = len(corpus_nightsII.words(file))

sentence_dic_nights = {}
for file in corpus_nightsII.fileids(): #calculate the amount of sentences in each volume
	sentence_dic_nights[file] = len(corpus_nightsII.sents(file))



