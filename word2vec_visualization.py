import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import gensim


LOGDIR = 'log'
N = 3000
D = 300
# load model
word2vec = gensim.models.KeyedVectors.load_word2vec_format('filtered.bin', binary=True)
vocab = list(word2vec.vocab.keys())
# create a list of vectors
embedding = np.empty((N, D), dtype=np.float32)
for i, word in enumerate(word2vec.index2word[: N]):
    embedding[i] = word2vec[word]

# setup a TensorFlow session
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=embedding.shape)
set_x = tf.assign(X, place, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embedding})

# write labels
with open('log/metadata.tsv', 'w') as f:
    for word in vocab[:N]:
        f.write(word + '\n')

# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = X.name
embedding_conf.metadata_path = os.path.join(LOGDIR, 'metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

# save the model
saver = tf.train.Saver()
saver.save(sess, os.path.join('log', "model.ckpt"))


#chekiranje za github
