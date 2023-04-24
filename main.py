# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:08:46 2018

@author: HJUNG
"""
import NOTAM_API
import wordfilter

stemft = False

qlist = []
plist = []
alltext = []
wlist = []

Statelist = ['USA', 'CAN', 'DEU', 'FRA', 'GBR', 'ESP', 'RUS', 'JPN', 'KOR', 'CHN', 'IND',
             'THA', 'MYS', 'TUR', 'BRA', 'MEX', 'IDN', 'ARE', 'AUS', 'BEL', 'ARG', 'BGD',
             'CMR', 'CHL', 'CUB', 'DNK', 'EGY', 'ETH', 'GRC', 'IRN', 'IRQ', 'IRL', 'KEN',
             'LBY', 'MAR', 'NLD', 'NGA', 'PHL', 'SAU', 'SGP', 'ZAF', 'SWE', 'UKR', 'VNM',
             'TUN', 'SYR', 'POL', 'PER', 'KAZ', 'HTI', 'CIV', 'GHA', 'GTM', 'PRK', 'ITA']
#Statelist = ['USA', 'CAN', 'DEU', 'FRA', 'GBR', 'ESP', 'RUS', 'JPN', 'KOR', 'CHN', 'IND']
#Statelist = ['USA', 'CAN']
for s in Statelist:
    Data, error = NOTAM_API.getNOTAM(api_key='f2f818c0-3d00-11e8-b03e-177c16a7d37a', states=s)
    
    if error is not None:
        print("Error:", error)
    
    
    for line in Data:
        text = line['all'].replace('\n', ' ').replace('.', '').lower()
        
        if ' q)' in text: 
            qcode = text.split(' q)', 1)[1].split('/')[1][1:]
        else:
            qcode = ''
            
            
        if ' e)' in text: 
            ptext = text.split(' e)', 1)[-1].rstrip()
        else:
            ptext = text
        
        # Full Text: TargetText = text, else: Targettext = ptext
#        ttext = text
        ttext = ptext
        
        spechar = ['/', '.', '(', ')', ':', '-', '!', ',', '=', '[', ']', '_', '?','{','}','*', '\'', '\"']
        for i in spechar:
            ttext = ttext.replace(i, ' ')
    
        temlst = ttext.split(' ')
#        print('p: ',ptext)
        if stemft: #use StemmingHelper filter
            for i in range(len(temlst)):
                temlst[i] = wordfilter.StemmingHelper.stem(temlst[i])
                if len(temlst[i]) > 1 and temlst[i].isdigit() == False:
                    wlist.append(temlst[i])
            ptext = ' '.join(word for word in temlst)
        else:
            for i in range(len(temlst)):
                if len(temlst[i]) > 1 and temlst[i].isdigit() == False:
                    wlist.append(temlst[i])
#        print('t: ',temlst)
        
        plist.append(ptext)
        qlist.append(qcode)
        alltext.append(text)


import collections
import math
import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: set the data.

vocabulary = wlist
print('Data size', len(vocabulary), ' Uniquewords:', len(set(vocabulary)))

# Step 2: Build the dictionary and replace rare words with UNK token.
#vocabulary_size = int(len(set(vocabulary)) * 0.25)
vocabulary_size = int(len(set(vocabulary)) * 1)

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 1024
embedding_size = 1024  # Dimension of the embedding vector.
skip_window = 5       # How many words to consider left and right.
num_skips = 4         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_window = 150  # Only pick dev samples in the head of the distribution.

#valid_examples = np.random.choice(valid_window, valid_size, replace=False)
lookup = ['apr', 'rwy', 'agl', 'twy', 'vor', 'out', 'ft', 'aip', 'sfc', 'ad', 'deg', 'acft', 'aircraft', 'obst', 'ils', 'ats', 'lgt',
          'airspace', 'amend', 'day', 'route', 'rnav', 'radius', 'nm', 'cat', 'tower', 'wind', 'enr',
          'iap', 'crane', 'gnd', 'gps', 'chart', 'avbl', 'procedure', 'departure', 'approach', 'airport', 'apch',
          'following', 'restricted', 'intl', 'ref', 'end', 'sup', 'circling', 'visibility', 'north', 'notam',
          'freq', 'arp', 'below', 'flight', 'rvr', 'atc', 'ifr', 'ramp', 'shall', 'by', 'at', 'to', 'the']


valid_examples = np.zeros(len(lookup))
for i in range(len(lookup)):
    valid_examples[i] = dictionary.get(lookup[i])


valid_size = len(lookup)     # Random set of words to evaluate similarity on.
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
#    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    learning_rate = tf.placeholder(tf.float32, shape=[])

    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# Step 5: Begin training.
num_steps = 5000001

with tf.Session(graph=graph) as session:
    saver = tf.train.Saver()
  # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        
        if step % 2000 == 0:
            ebd = normalized_embeddings.eval()
            if step > 0:
                average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0
    
        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
    
    
#    for i in range(vocabulary_size):
#        embed = final_embeddings[i, :]
#        word = dictionary
#        file_.write('%s %s\n' % (word, ' '.join(map(str, embed))))
    
    saver.save(session, 'my_test_model')
#    saver.restore(session, "/model.ckpt")
# Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(50, 50))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    axes = plt.gca()
    axes.set_xlim([-75,75])
    axes.set_ylim([-75,75])
    plt.show()
#        plt.savefig(filename)

try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = min(2000, vocabulary_size)
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')

