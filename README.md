Text Analysis of 1001 Nights (with Latent Dirichlet Allocation)

In this project, we perform a text analysis on the tales of 1001 Nights.

Basic statistics such as the readability score of a tale, the amount of sentences, words... were calculated for each file and visualised by using matplotlib and numpy. A deeper analysis was performed by using the ldamodel of gensim, a topic modeling module. The model was then trained on a corpus of tales and tested on our 1001 nights. Afterwards, topic richness was computed and the documents were hierarchically clustered into various dendograms. Clustered topics were also visualized in word clouds. Lastly, topics were visualised by colouring the words of a tale per topic. 

