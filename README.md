# Text-classification-CNN-vord2wec
Sentiment analysis of movie review with word2vec and convolutional neural network

First approach to this task is by averaging vectors (more about this approach can be found here http://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis) using Logistic regression algorithm. The idea is to use already pretrained word2vec google news model, search for every word in review in that model for vector and use these vectors as an input to algorithm. I this approach, we use just one average vector of of review. If dataset is consisted of 1000 reviwes, we are gonna have 1000 inputs (vectors) that we seperate into training and testing data. Results are good but within this dataset that i have been used, this approach is not good because reviews are long and we loose lots of information by averaging.

Second approach is that we use every word from review as a vector (we make some kind of "image" of vectors for CNN) and CNN is fed by these embeddings. Results are amazing. Depends on how much data do you have, the architecture of CNN and so on. 
