from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np
from sklearn.cross_validation import train_test_split
from SentimentClassifier import SentimentClassifier
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def generate_feature_vectors(doc, model):
 
    vec = np.zeros(300).reshape((1, 300))  
    count = 0
    for word in doc.split():
        if model.__contains__(word.strip()):
            count = count + 1
            vec += model[word.strip()]
    vec = vec / count
    return vec


def read_dataset(pos_dataset, neg_dataset):

    with open(pos_dataset, 'r') as pos_reviews:
            pos_reviews = pos_reviews.readlines()
             
    with open(neg_dataset, 'r') as neg_reviews:
            neg_reviews = neg_reviews.readlines()
        
    return (pos_reviews, neg_reviews)



def split_dataset_train_test(pos_dataset, neg_dataset, pred_values, test_prop):
    
    features_train, features_test, class_train, class_test = train_test_split(np.concatenate((pos_dataset, neg_dataset)), pred_values, test_size=test_prop)
    return (features_train, features_test, class_train, class_test)



def generate_features(word2vec_model, data):
    
    features = np.concatenate([generate_feature_vectors(s, model) for s in data])
    return features

def split_dataset_train_test(pos_dataset, neg_dataset, pred_values, test_prop):
    
    features_train, features_test, class_train, class_test = train_test_split(np.concatenate((pos_dataset, neg_dataset)), pred_values, test_size=test_prop)
    return (features_train, features_test, class_train, class_test)


pos_reviews_file = 'positive.txt'
neg_reviews_file = 'negative.txt'


model = gensim.models.KeyedVectors.load_word2vec_format('filtered.bin', binary=True)
print('word2vec sucessfully ')
positive_reviews, negative_reviews = read_dataset(pos_reviews_file, neg_reviews_file)
print('Post and neg data have been sucessfully uploaded')
print(len(positive_reviews))
positive_reviews = positive_reviews[:1000]
negative_reviews = negative_reviews[:1000]



#create an array containing the values of the predicted variable for training
y = np.concatenate((np.ones(len(positive_reviews)), np.zeros(len(negative_reviews))))
#split the dataset into training and testing
x_train, x_test, y_train, y_test = split_dataset_train_test(positive_reviews, negative_reviews, y, 0.2)
print("************************************",type(x_train))

#generate features for the training set    
training_vectors = generate_features(model, x_train	)
test_vectors = generate_features(model, x_test)
#print(training_vectors[0])
#print(type(training_vectors))
#print(training_vectors.shape)

#print(np.isnan(training_vectors).any())


sc = SentimentClassifier()
sc.fit(training_vectors, y_train)
print("Training process completes")

pred_probs = sc.predict(test_vectors)
print("Testing process completes")
pred_class = [1 if ele > 0.5 else 0 for ele in pred_probs]
recall = recall_score(y_test, pred_class)
print ("The recall score for the positive sentiment is %s percent" % str(recall*100.0))

precision = precision_score(y_test, pred_class)
print ("The precision score for the positive sentiment is %s percent" % str(precision*100.0))

accuracy = accuracy_score(y_test, pred_class)
print ("The accruacy score of the classifier is %s percent" % str(accuracy*100.0))


