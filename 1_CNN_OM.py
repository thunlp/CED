from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.contrib import layers
import tensorflow as tf
import numpy as np
import functools
import codecs
import random
import json
import time
import math
import os

# Model and training hyper-params
train_sz = 5435
verify_sz = 800

test_sz = 1810

epoch_num = 10
batch_sz = 5
max_len = 300
K = 1000

vocabulary_size = 20000
embedding_size = 200
weibo_len = 30

num_checkpoints = 5

data_redio = 1.0

data = tf.placeholder(tf.int32, [None, weibo_len])
target = tf.placeholder(tf.float32, [None, 2])

global_step = tf.Variable(0, name="global_step", trainable=False)

def lazy_property(function):
    attribute = '_' + function.__name__
    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class VariableSequenceClassification:

    def __init__(self, data, target, num_filters=50, filter_sizes=(2, 3, 4, 5)):

        self.data = data
        self.target = target
        self.num_classes = int(self.target.get_shape()[1])
        self.l2_reg_lambda = 0.002
        self.cur_seed = random.getrandbits(64)

        # Add regular item
        self.l2_loss = tf.constant(0.0)

        # cnn
        # embedding
        self.W = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="W")
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.data)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, mean=1.0), name="W")
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, weibo_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # combine all the pooled features
        self.num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        #self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])


        self.weight = tf.get_variable(name="weight", shape=[self.num_filters_total, self.num_classes],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.cur_seed))
        self.bias = tf.get_variable(name="bias", shape=[self.num_classes],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.cur_seed))

        self.l2_loss += tf.nn.l2_loss(self.weight)
        self.l2_loss += tf.nn.l2_loss(self.bias)

        self.error
        self.cnn_optimize
        self.cnn_detection

    @lazy_property
    def cnn_cost(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.cnn_detection_no_softmax, labels=self.target)
        return tf.reduce_sum(loss) + self.l2_reg_lambda * self.l2_loss

    @lazy_property
    def cnn_optimize(self):
        learning_rate = 0.001
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cnn_cost, global_step=global_step)

    @lazy_property
    def cnn_detection(self):
        prediction = self.cnn_detection_no_softmax
        prediction = tf.nn.softmax(prediction)
        return prediction

    @lazy_property
    def cnn_detection_no_softmax(self):
        h = self.h_pool_flat
        prediction = tf.nn.xw_plus_b(h, self.weight, self.bias)
        #prediction = tf.nn.softmax(prediction)
        return prediction

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.cnn_detection, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @lazy_property
    def detection_result(self):
        result = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.cnn_detection, 1))
        return result

class DataTensor:
    def __init__(self, namelist):
        length = len(namelist)
        self.data = np.zeros((length, weibo_len))
        self.tensor = np.zeros((length, 2))

        for j in range(length):
            name = namelist[j]
            self.tensor[j, :] = class_dic[name]['class']
            psg = np.array(msg_seq_vec_dic[name])
            len_psg = len(psg)
            len_psg = int(len_psg * data_redio)
            self.data[j, :len_psg] = psg[:len_psg]

    def feed(self):
        return {data: self.data, target: self.tensor}

target
if __name__ == '__main__':

    with codecs.open("msg_id.txt", "r", 'utf-8') as f:
        msg_seq_vec_dic = json.load(f, encoding='utf-8')
    with codecs.open("class_8050.json", "r", 'utf-8') as f:
        class_dic = json.load(f, encoding='utf-8')
    print 'file read'

    # handle all zero vectors
    broken = []
    for name, psg in msg_seq_vec_dic.iteritems():
        #if class_dic[name]['len'] < 20 or psg == []:
        if psg == []:
            broken.append(name)
    for name in broken:
        msg_seq_vec_dic.pop(name)
    print 'broken handled'

    keys = msg_seq_vec_dic.keys()
    random.seed(32)
    random.shuffle(keys)

    train_names = keys[0:train_sz]
    verify_names = keys[train_sz:train_sz+verify_sz]
    test_names = keys[train_sz+verify_sz:train_sz + verify_sz + test_sz]
    print 'train_num:', len(train_names)
    print 'verify_num', len(verify_names)
    print 'test_num:', len(test_names)
    # exit(0)

    model = VariableSequenceClassification(data, target)

    sess = tf.Session()

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "1_CNN_OM_runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

    sess.run(tf.global_variables_initializer())

    test = DataTensor(test_names)
    verify = DataTensor(verify_names)
    train = DataTensor(train_names)

    test_true = np.argmin(test.tensor, 1)
    np.array(test_true)
    print test_true
    train_true = np.argmin(train.tensor, 1)
    np.array(train_true)
    verify_true = np.argmin(verify.tensor, 1)
    np.array(verify_true)

    print 'Test and Train DataTensor Done'

    max_accu = 0
    best_at_step = 0

    for epoch in range(epoch_num):
        for i in range(train_sz/batch_sz):
            batch = DataTensor(train_names[i*batch_sz:(i+1)*batch_sz])
            sess.run(model.cnn_optimize, batch.feed())   # learning here
            current_step = tf.train.global_step(sess, global_step)
            #print current_step

            if current_step % 100 == 0:
                prediction = []
                for i in range(verify_sz/batch_sz):
                    verify_batch = DataTensor(verify_names[i*batch_sz:(i+1)*batch_sz])
                    detection_result = sess.run(model.cnn_detection, verify_batch.feed())
                    prediction.extend(detection_result.tolist())
                verify_pred = np.argmin(prediction, 1)
                np.array(verify_pred)
                verify_accu = accuracy_score(verify_true, verify_pred)
                if verify_accu >= max_accu:
                    max_accu = verify_accu
                    best_at_step = current_step
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print 'Best of valid = {}, at step {}'.format(max_accu, best_at_step)

            if i % 10 == 0:
                train_error = sess.run(model.error, batch.feed())
                train_cost = sess.run(model.cnn_cost, batch.feed())
                train_pred = sess.run(model.cnn_detection, batch.feed())
                #check_weibo = sess.run(model.h_pool_flat, batch.feed())
                #check_weibo_shape = sess.run(model.weibo_shape, batch.feed())
                print('train: Epoch {:1d} Batch {:2d} error {:3.1f}%'.format(epoch + 1, i + 1, 100 * train_error))
                print('train: Loss {:3.1f}'.format(train_cost))
                #print check_weibo
                print train_pred
                #print check_weibo_shape

        right_one = 0
        for i in range(verify_sz/batch_sz):
            verify_batch = DataTensor(verify_names[i*batch_sz:(i+1)*batch_sz])
            detection_result = sess.run(model.detection_result, verify_batch.feed())
            detection_result = detection_result.tolist()
            true_num = detection_result.count(True)
            right_one = right_one + true_num
        verify_accuracy = float(right_one)/verify_sz
        print('VERIFY: Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * verify_accuracy))

        train_er = []
        train_right_one = 0
        train_prediction = []
        for i in range(train_sz/batch_sz):
            train_batch = DataTensor(train_names[i*batch_sz:(i+1)*batch_sz])

            # early_result = sess.run(model.early, train_batch.feed())
            # early_result = early_result.tolist()
            # if not math.isnan(early_result):
                # train_er.append(early_result)

            detection_result = sess.run(model.detection_result, train_batch.feed())
            detection_result = detection_result.tolist()
            true_num = detection_result.count(True)
            train_right_one = train_right_one + true_num

            detection_result = sess.run(model.cnn_detection, train_batch.feed())
            train_prediction.extend(detection_result.tolist())
        #early = sum(train_er)/len(train_er)
        #print('TRAIN: Early {:2.3f}%'.format(100 * early))
        train_accuracy = float(train_right_one)/train_sz
        print('TRAIN: Error {:2.3f}%'.format(100 * train_accuracy))
        train_pred = np.argmin(train_prediction, 1)
        np.array(train_pred)
        Accu = accuracy_score(train_true, train_pred)
        print('TRAIN: Accuracy {:2.3f}%'.format(100 * Accu))

        verify_er = []
        verify_right_one = 0
        verify_prediction = []
        for i in range(verify_sz/batch_sz):
            verify_batch = DataTensor(verify_names[i*batch_sz:(i+1)*batch_sz])

            # early_result = sess.run(model.early, verify_batch.feed())
            # early_result = early_result.tolist()
            # if not math.isnan(early_result):
                # verify_er.append(early_result)

            detection_result = sess.run(model.detection_result, verify_batch.feed())
            detection_result = detection_result.tolist()
            true_num = detection_result.count(True)
            verify_right_one = verify_right_one + true_num

            detection_result = sess.run(model.cnn_detection, verify_batch.feed())
            verify_prediction.extend(detection_result.tolist())
        #early = sum(verify_er)/len(verify_er)
        #print('VERIFY: Early {:2.3f}%'.format(100 * early))
        verify_accuracy = float(verify_right_one)/verify_sz
        print('VERIFY: Error {:2.3f}%'.format(100 * verify_accuracy))
        verify_pred = np.argmin(verify_prediction, 1)
        np.array(verify_pred)
        Accu = accuracy_score(verify_true, verify_pred)
        print('VERIFY: Accuracy {:2.3f}%'.format(100 * Accu))

        error = sess.run(model.error, verify.feed())
        print('VERIFY: Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))


    saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
    print 'Finish training. On test set:'

    test_er = []
    right_one = 0
    prediction = []
    for i in range(test_sz/batch_sz):
        test_batch = DataTensor(test_names[i*batch_sz:(i+1)*batch_sz])

        # early_result = sess.run(model.early, test_batch.feed())
        # early_result = early_result.tolist()
        # if not math.isnan(early_result):
            # test_er.append(early_result)

        detection_result = sess.run(model.detection_result, test_batch.feed())
        detection_result = detection_result.tolist()
        true_num = detection_result.count(True)
        right_one = right_one + true_num

        detection_result = sess.run(model.cnn_detection, test_batch.feed())
        prediction.extend(detection_result.tolist())

    #early = sum(test_er)/len(test_er)
    #print('TEST: Early {:2.3f}%'.format(100 * early))
    test_accuracy = float(right_one)/test_sz
    print('TEST: Error {:2.3f}%'.format(100 * test_accuracy))
    print prediction

    test_pred = np.argmin(prediction, 1)
    np.array(test_pred)
    print test_pred
    P = precision_score(test_true, test_pred)
    R = recall_score(test_true, test_pred)
    Accu = accuracy_score(test_true, test_pred)
    f1score = f1_score(test_true, test_pred, average='binary')
    print 'Accu', Accu
    print 'P', P
    print 'R', R
    print 'F1', f1score