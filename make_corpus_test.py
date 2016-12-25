<<<<<<< HEAD
#from nltk.corpus import PlaintextCorpusReader #maak corpus
#corpus_root= 'data'
#wordlists = PlaintextCorpusReader(corpus_root, '.*')
#new_corpus = wordlists.fileids()
=======
from nltk.corpus import PlaintextCorpusReader #maak corpus
corpus_root= 'data'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
corpus = wordlists.fileids()
print(corpus)
print(wordlists.words('arabian_nights_burton_vol01.txt'))
print(wordlists.sents('arabian_nights_burton_vol01.txt')[0:20])




>>>>>>> origin/master
#print(new_corpus)
#wordlists.words('file_name')--> of .sents om aparte woorden of zinnen per file te krijgen


#from nltk.corpus import stopwords #clean corpus
#from nltk.stem.wordnet import WordNetLemmatizer
#import string

#import gensim
#from gensim import corpora
<<<<<<< HEAD


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

print(calculate_characters(corpus))	#geeft none?

#make a list of the total characters per volume ---> lijst maken van de counters die def(calculate_characters) teruggeeft
characters_per_volume= []
characters_per_volume.append(calculate_characters(corpus)) #ik weet even niet goed hoe ik hier een lijst van kan maken, dit klopt niet
print(characters_per_volume)

 
import matplotlib.pyplot as plt #pyhton library for plotting data
import numpy as np
#%matplotlib inline
characters_per_volume = [719626, 713975, 740002, 608981, 805537, 620786, 816219, 728202, 743379, 130715] #nu heb ik handmatig een lijst gemaakt
 #van de aantallen characters per volume
x = [1,2,3,4,5,6,7,8,9,10]
y = characters_per_volume
x_labels = ["vol1","vol2","vol3","vol4","vol5","vol6","vol7","vol8","vol9","vol10"]
y_labels = [719626, 713975, 740002, 608981, 805537, 620786, 816219, 728202, 743379, 130715]
width = 0.12 # width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x, y_labels)
def autolabel(rects):# attach labels above bars #zoals gevonden op matplotlib site
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom') #The placement of the text is determined by the height function, or the height of the column
        #and the number that is put on top of each column is written by: '%d' %int(height). 
        #So all you need to do is create an array of strings, called 'name', that you want at the top of the columns and iterate through. 

print(autolabel(rects1))
plt.bar(x,y)
plt.title("total characters in one volume")
plt.xlabel("volume")
plt.ylabel("# of characters")
plt.xticks(x,x_labels)
plt.show()
=======
>>>>>>> origin/master
