import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import gensim
from sklearn.model_selection import train_test_split

positive_data = 'positive.txt'
negative_data = 'negative.txt'
word2vec_model_path = 'filtered.bin'
word2vec_binary = True


print('Loading data...')
dataset = data_helpers.get_datasets_mrpolarity(positive_data, negative_data)
print('Data successfully laded!\n')

print('Making labels...')
x_data, y = data_helpers.load_data_labels(dataset)

print(len(y))

max_document_length = max([len(x.split(" ")) for x in x_data])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_data)))


#shuffling data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

#making training and testing data, 20% for testing
x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.2)

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))



seq_length = x_train.shape[1]
number_of_classes = y_train.shape[1]
vocabulary_size = len(vocab_processor.vocabulary_)
embedding_dimension = 300
filters = 2, 3, 4
num_filters = 128
l2_reg_lambda = 0.0
dropout_keep_prob = 0.5
batch_size = 64
num_epochs = 50
evaluate_every = 25
l2_reg_lambda = 0.0
vocabulary = vocab_processor.vocabulary_
#Training process



with tf.Graph().as_default():
    
    sess = tf.Session()
    with sess.as_default():
        cnn = TextCNN(
            sequence_length = seq_length, num_classes = number_of_classes, vocab_size = vocabulary_size,
            embedding_size = embedding_dimension, filter_sizes = filters, num_filters = num_filters, l2_reg_lambda = l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        initW = None
        # load embedding vectors from the word2vec
        initW = data_helpers.load_embedding_vectors_word2vec(vocabulary, word2vec_model_path, word2vec_binary)
        print("word2vec file has been loaded")
        sess.run(cnn.W.assign(initW))


        def train_step(x_batch, y_batch):

            feed_dict = { cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: dropout_keep_prob}
            _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def dev_step(x_batch, y_batch, writer=None):
  
            feed_dict = { cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0 }
            step, loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("Testing... ")
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))


        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)
        # Training loop. For each batch...

        '''
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            
        dev_step(x_dev, y_dev)
        print("DONE!")
		'''

        for batch in batches:
	        x_batch, y_batch = zip(*batch)
	        train_step(x_batch, y_batch)
	        current_step = tf.train.global_step(sess, global_step)
	        
        print("\nEvaluation:")
        dev_step(x_dev, y_dev)
        print("")




