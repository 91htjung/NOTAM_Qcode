# -*- coding: utf-8 -*-
"""
Created on Mon May 14 15:27:07 2018

@author: hjung
"""


import os
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.keras.preprocessing import sequence
from tensorboard import summary as summary_lib

model_dir = os.curdir
df = pd.read_csv('NOTAMs.csv', engine='python')
df = df.dropna()

#filter non-standard NOTAMS
fdf = df[(df['feature1'] != 'XX') & (df['feature2'] != 'XX')]
xdf = df[(df['feature1'] == 'XX') | (df['feature2'] == 'XX')]

#elst = ['LA', 'LB', 'LC', 'LD', 'LE', 'LF', 'LG', 'LH', 'LI', 'LJ', 'LK', 'LL', 'LM', 'LP', 'LR', 'LS', 'LT', 'LU', 'LV', 'LW', 'LX', 'LY', 'LZ',
#        'MA', 'MB', 'MC', 'MD', 'MG', 'MH', 'MK', 'MM', 'MN', 'MO', 'MP', 'MR', 'MS', 'MT', 'MU', 'MW', 'MX', 'MY',
#        'FA', 'FB', 'FC', 'FD', 'FE', 'FF', 'FG', 'FH', 'FI', 'FJ', 'FL', 'FM', 'FO', 'FP', 'FS', 'FT', 'FU', 'FW', 'FZ',
#        'AA', 'AC', 'AD', 'AE', 'AF', 'AH', 'AL', 'AN', 'AO', 'AP', 'AR', 'AT', 'AU', 'AV', 'AX', 'AZ',
#        'SA', 'SB', 'SC', 'SE', 'SF', 'SL', 'SO', 'SP', 'SS', 'ST', 'SU', 'SV', 'SY',
#        'PA', 'PB', 'PC', 'PD', 'PE', 'PF', 'PH', 'PI', 'PK', 'PL', 'PM', 'PN', 'PO', 'PR', 'PT', 'PU', 'PX', 'PZ',
#        'CA', 'CB', 'CC', 'CD', 'CE', 'CG', 'CL', 'CM', 'CP', 'CR', 'CS', 'CT',
#        'IC', 'ID', 'IG', 'II', 'IL', 'IM', 'IN', 'IO', 'IS', 'IT', 'IU', 'IW',
#        'IX', 'IY'
#        ]



e_cnt = fdf['feature1'].value_counts()
s_cnt = fdf['feature2'].value_counts()
#qcode_cnt = df['qcode'].value_counts()
#f1_cnt.plot(kind='bar', figsize=(20,20))
#f2_cnt.plot(kind='bar', figsize=(20,20))
#qcode_cnt.plot(kind='bar', figsize=(20,20))

e_index = {}
s_index = {}

for i in fdf['feature1'].unique():
    e_index[i] = len(e_index)+1

for i in fdf['feature2'].unique():
    s_index[i] = len(s_index)+1

fdf["feature1"].replace(e_index, inplace=True)
fdf["feature2"].replace(s_index, inplace=True)


# Should we not use keras and rewrite this logic?
print("Loading data...")

spechar = ['/', '(', ')', ':', '-', '!', ',', '=', '[', ']', '_', '?','{','}','*', '\'', '\"']
vocabulary = []
x = []
i = 0
for j in fdf['text'].tolist():
    if j is not None:
        
        if ' e)' in j: 
            j = j.split(' e)', 1)[-1].rstrip()
        else:
            j = j
            
        for jj in spechar:
            j = j.replace(jj, ' ')
        
        
        jlst = j.split(' ')
        jjlst = []
        for k in range(len(jlst)):
            
            if len(jlst[k]) > 1 and jlst[k].isdigit() == False:
#                if jlst[k] not in vocabulary:
                vocabulary.append(jlst[k])
                jjlst.append(jlst[k])
        
        x.append(jjlst)
#    else:
#        print(i)
#        x.append(None)
    
    i = i + 1

print("initial data loaded", len(x))

vocabulary_size = int(len(set(vocabulary)) * 1)
embedding_size = 1024

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

data, count, word_index, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)

print("vocabulary:", len(vocabulary), " unique vocab: ", vocabulary_size)


print("setting word vector")
# word vector x
for ii in range(len(x)):
    words = x[ii]
    for i in range(len(words)):
        if words[i] in word_index:
            x[ii][i] = word_index[words[i]]
        else:
            x[ii][i] = word_index['UNK']

x = np.asarray(x, dtype=object)


print("setting label")
# labeling y
label_index = {}
label_count = {}
qlst = fdf['qcode'].tolist()
y = []
for j in qlst:
    if j not in label_count.keys():
#        label_index[j] = len(label_index)
        label_count[j] = 0
#    y.append(label_index[j])
    label_count[j] = label_count[j] + 1


# filter qcode less than occurrence
min_occur = 5

delind = []
for j in range(len(qlst)):
    if label_count[qlst[j]] >= min_occur:
        if qlst[j] not in label_index.keys():
            label_index[qlst[j]] = len(label_index)
        
        y.append(label_index[qlst[j]])
    else:
        delind.append(j)
        
reversed_label = dict(zip(label_index.values(), label_index.keys()))

y = np.asarray(y, dtype=np.int32)
x = np.delete(x,delind)




print("data parsed into..", len(x), "/", len(y))

# import pre-trained embeddings
with open('embedding.pkl', 'rb') as f:
    imp_embd = pickle.load(f)

with open('dict.pkl', 'rb') as f:
    imp_vocab = pickle.load(f)

embedding_matrix = np.random.uniform(-1, 1, size=(vocabulary_size, embedding_size))

num_loaded = 0
for w, i in word_index.items():
    v = imp_vocab.get(w)
    if v is not None and i < vocabulary_size:
        embedding_matrix[i] = imp_embd[v]
        num_loaded += 1
        
print('Successfully loaded pretrained embeddings for '
          f'{num_loaded}/{vocabulary_size} words.')

embedding_matrix = embedding_matrix.astype(np.float32)

## Embedding visualization
    
sampleseed = 1
samplesize = 0.8

np.random.seed(sampleseed)

indices = np.random.permutation(y.shape[0])
#indices = np.random.permutation(len(y))

training_idx, test_idx = indices[:int(samplesize * len(y))], indices[int(samplesize * len(y)):]
x_train_variable, x_test_variable = x[training_idx], x[test_idx]
y_train, y_test = y[training_idx], y[test_idx]

y_test = y_test.tolist()
y_train = y_train.tolist()

print(len(y_train), "train sequences")
print(len(y_test), "test sequences")


sentence_size = 100


print("Pad sequences (samples x time)")
x_train = sequence.pad_sequences(x_train_variable, 
                                 maxlen=sentence_size, 
                                 padding='post', 
                                 value=0)
x_test = sequence.pad_sequences(x_test_variable, 
                                maxlen=sentence_size, 
                                padding='post', 
                                value=0)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

"""We can use the word index map to inspect how the first review looks like."""

word_inverted_index = {v: k for k, v in word_index.items()}

def index_to_text(indexes):
    return ' '.join([word_inverted_index[i] for i in indexes])

print(index_to_text(x_train_variable[0]))

"""## Building Estimators
In the next section we will build several models to make predictions for the labels in the dataset. We will first use canned estimators and then create custom ones for the task. We recommend that you check out [this blog post](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html) that explains how to use the `tf.feature_column` module to standardize and abstract how raw input data is processed and [the following one](https://developers.googleblog.com/2017/12/creating-custom-estimators-in-tensorflow.html) that covers in depth how to work with Estimators.
### From arrays to tensors
There's one more thing we need to do get our data ready for TensorFlow. We need to convert the data from numpy arrays into Tensors. Fortunately for us the `Dataset` module has us covered. 
It provides a handy function, `from_tensor_slices` that creates the dataset to which we can then apply multiple transformations to shuffle, batch and repeat samples and plug into our training pipeline. Moreover, with just a few changes we could be loading the data from files on disk and the framework does all the memory management.
We define two input functions: one for processing the training data and one for processing the test data. We shuffle the training data and do not predefine the number of epochs we want to train, while we only need one epoch of the test data for evaluation. We also add an additional `"len"` key to both that captures the length of the original, unpadded sequence, which we will use later.
"""

x_len_train = np.array([min(len(x), sentence_size) for x in x_train_variable])
x_len_test = np.array([min(len(x), sentence_size) for x in x_test_variable])

def parser(x, length, y):
    features = {"x": x, "len": length}
    return features, y

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_train, x_len_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(x_train_variable))
    dataset = dataset.batch(244)
    dataset = dataset.map(parser)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_test, x_len_test, y_test))
    dataset = dataset.batch(244)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

"""### Baselines
It's always a good practice to start any machine learning project trying out a couple of reliable baselines. Simple is always better and it is key to understand exactly how much we are gaining in terms of performance by adding extra complexity. It may very well be the case that a simple solution is good enough for our requirements.
With that in mind, let us start by trying out one of the simplest models out there for text classification. That is, a sparse linear model that gives a weight to each token and adds up all of the results, regardless of the order. The fact that we don't care about the order of the words in the sentence is the reason why this method is generally known as a Bag-of-Words (BOW) approach. Let's see how that works out.
We start out by defining the feature column that is used as input to our classifier. As we've seen [in this blog post](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html), `categorical_column_with_identity` is the right choice for this pre-processed text input. If we were feeding raw text tokens, other `feature_columns` could do a lot of the pre-processing for us. We can now use the pre-made `LinearClassifier`.
"""

#column = tf.feature_column.categorical_column_with_identity('x', vocabulary_size)
#classifier = tf.estimator.LinearClassifier(feature_columns=[column], model_dir=os.path.join(model_dir, 'bow_sparse'), label_vocabulary=np.unique(y).tolist())


"""Finally, we create a simple function that trains the classifier and additionally creates a precision-recall curve. Note that we do not aim to maximize performance in this blog post, so we only train our models for $25,000$ steps."""

all_classifiers = {}
def train_and_evaluate(classifier):
    # Save a reference to the classifier to run predictions later
    all_classifiers[classifier.model_dir] = classifier
    classifier.train(input_fn=train_input_fn, steps=25000)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    predictions = np.array([p['logistic'][0] for p in classifier.predict(input_fn=eval_input_fn)])
        
    # Reset the graph to be able to reuse name scopes
    tf.reset_default_graph() 
    # Add a PR summary in addition to the summaries that the classifier writes
    pr = summary_lib.pr_curve('precision_recall', predictions=predictions, labels=y_test, num_thresholds=21)
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(os.path.join(classifier.model_dir, 'eval'), sess.graph)
        writer.add_summary(sess.run(pr), global_step=0)
        writer.close()
#     # Un-comment code to download experiment data from Colaboratory
#     from google.colab import files
#     model_name = os.path.basename(os.path.normpath(classifier.model_dir))
#     ! zip -r {model_name + '.zip'} {classifier.model_dir}
#     files.download(model_name + '.zip')


head = tf.contrib.estimator.multi_label_head(n_classes=244)

def cnn_model_fn(features, labels, mode, params):    
    input_layer = tf.contrib.layers.embed_sequence(
        features['x'], vocabulary_size, embedding_size,
        initializer=params['embedding_initializer'])
    
    
    training = mode == tf.estimator.ModeKeys.TRAIN
    dropout_emb = tf.layers.dropout(inputs=input_layer, 
                                    rate=0.2, 
                                    training=training)
    
    
    conv1 = tf.layers.conv1d(
        inputs=dropout_emb,
        filters=32,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)
    
    # Global Max Pooling
    pool1 = tf.reduce_max(input_tensor=conv1, axis=1)
#    pool = tf.reduce_max(input_tensor=conv)
    
#    pool1_flat = tf.reshape(pool1, [-1])
    dense = tf.layers.dense(inputs=pool1, units=500, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=training)
    
    # Logits Layer
#    flat_dropout = tf.reshape(tensor=dropout, shape=(-1, 500))
    logits = tf.layers.dense(inputs=dropout, units=params['n_classes'])

#    flat_logits = tf.reshape(tensor=logits, shape=(-1,params['n_classes']))
#    hidden = tf.layers.dense(inputs=pool, units=500, activation=tf.nn.relu)
#    
#    dropout_hidden = tf.layers.dropout(inputs=hidden, 
#                                       rate=0.2, 
#                                       training=training)
    
#    logits = tf.layers.dense(inputs=dropout_hidden, units=params['n_classes'])
    
#     This will be None when predicting
    if labels is not None:
        labels = tf.reshape(labels, [-1, params['n_classes']])
    
#    flat_labels = tf.reshape(tensor=labels, shape=(-1,params['n_classes']))
    # Get the loss
#    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    
#    learning_rate = 0.001
#    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
#    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    labels = tf.cast(labels, tf.float32)
    loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), axis=1))
#    optimizer = tf.train.AdamOptimizer().minimize(loss)
    optimizer = tf.train.AdamOptimizer()
    
    
    def _train_op_fn(loss):
        return optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

    return head.create_estimator_spec(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits, 
        train_op_fn=_train_op_fn)
  

"""To create a CNN classifier that leverages pretrained embeddings, we can reuse our `cnn_model_fn` but pass in a custom initializer that initializes the embeddings with our pretrained embedding matrix."""

def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
    assert dtype is tf.float32
    return embedding_matrix




cnn_pretrained_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                        model_dir=os.path.join(model_dir, 'cnn_pretrained'),
                                        params={'embedding_initializer': my_initializer,
                                                # Two hidden layers of 10 nodes each.
#                                                'hidden_units': [10, 10],
                                                # The model must choose between 3 classes.
                                                'n_classes': 244})

#classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
#                                          hidden_units=[10, 20, 10],
#                                          n_classes=numClasses,
#                                          model_dir=saveAt,
#                                          label_vocabulary=uniqueTrain)
    
    
#params = {'embedding_initializer': my_initializer, 'n_classes':len(np.unique(y).tolist())}

#estimator = tf.estimator.DNNClassifier(
#      hidden_units=[500, 100],
#      feature_columns=x,
#      n_classes=2,
#      optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))    


print("start training")
train_and_evaluate(cnn_pretrained_classifier)








# Training for 1,000 steps means 128,000 training examples with the default
# batch size. This is roughly equivalent to 5 epochs since the training dataset
# contains 25,000 examples.
estimator.train(input_fn=train_input_fn, steps=1000);

train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))