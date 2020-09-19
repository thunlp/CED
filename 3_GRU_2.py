from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
import numpy as np
import functools
import codecs
import json
import random
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

data_redio = 1.0

num_checkpoints = 5

data = tf.placeholder(tf.float32, [None, max_len, K])
data_len = tf.placeholder(tf.int32, [None])
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

    def __init__(self, data, data_len, target, num_hidden=200, num_layers=2):
        self.data = data
        self.data_len = data_len
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.cur_seed = random.getrandbits(32)

        self.prediction
        self.error
        self.optimize

    @lazy_property
    def length(self):
        length = self.data_len
        length = tf.cast(length, tf.float32)
        #redio = tf.constant(data_redio, dtype=tf.float32, shape=[1, batch_sz])
        length = tf.floor(tf.multiply(length, data_redio))
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        # Recurrent network.
        output, _ = tf.nn.dynamic_rnn(
            tf.contrib.rnn.GRUCell(self._num_hidden),
            data,
            dtype=tf.float32,
            sequence_length=self.length,
        )
        # Using the last step output of RNN to predict
        last = self._last_relevant(output, self.length)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]), self.cur_seed)
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def detection_result(self):
        result = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return result

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost, global_step=global_step)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size, seed):
        weight = tf.get_variable(name="weight", shape=[in_size, out_size],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=seed))
        bias = tf.get_variable(name="bias", shape=[out_size],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed))
        return weight, bias

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

class DataTensor:
    def __init__(self, namelist):
        length = len(namelist)
        self.len = np.zeros(length, dtype=np.int32)
        self.tensor = np.zeros((length, 2))
        self.data = np.zeros((length, max_len, K))
        for j in range(length):
            name = namelist[j]
            psg = np.array(seq_vec_dic[name])
            self.len[j] = len(seq_vec_dic[name])
            self.tensor[j, :] = class_dic[name]['class']
            if len(psg) > max_len:
                temp = psg[:max_len, :]
                #print temp.shape
                self.data[j, :max_len, :] = temp
            else:
                #print psg.shape
                self.data[j, :len(psg), :] = psg

    def feed(self):
        return {data: self.data, data_len: self.len, target: self.tensor}

if __name__ == '__main__':
    # We treat images as sequences of pixel rows.
    with codecs.open("10_parted_posts_seqvec.txt", "r", 'utf-8') as f:
        seq_vec_dic = json.load(f, encoding='utf-8')
    with codecs.open("class_8050.json", "r", 'utf-8') as f:
        class_dic = json.load(f, encoding='utf-8')
    print 'file read'

    # handle all zero vectors
    broken = []
    for name, psg in seq_vec_dic.iteritems():
        #if class_dic[name]['len'] < 20 or psg == []:
        if psg == []:
            broken.append(name)
    for name in broken:
        seq_vec_dic.pop(name)
    print 'broken handled'

    keys = seq_vec_dic.keys()
    random.seed(32)
    random.shuffle(keys)

    train_names = keys[0:train_sz]
    verify_names = keys[train_sz:train_sz+verify_sz]
    test_names = keys[train_sz+verify_sz:train_sz + verify_sz + test_sz]
    print 'train_num:', len(train_names)
    print 'verify_num', len(verify_names)
    print 'test_num:', len(test_names)
    # exit(0)

    model = VariableSequenceClassification(data, data_len, target)

    sess = tf.Session()

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "3_GRU2_runs", timestamp))
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

    # Training
    for epoch in range(epoch_num):
        for i in range(train_sz/batch_sz):
            batch = DataTensor(train_names[i*batch_sz:(i+1)*batch_sz])
            sess.run(model.optimize, batch.feed())  # learning here
            current_step = tf.train.global_step(sess, global_step)
            #print current_step

            if current_step % 100 == 0:
                prediction = []
                for i in range(verify_sz/batch_sz):
                    verify_batch = DataTensor(verify_names[i*batch_sz:(i+1)*batch_sz])
                    detection_result = sess.run(model.prediction, verify_batch.feed())
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
                train_cost = sess.run(model.cost, batch.feed())
                train_pred = sess.run(model.prediction, batch.feed())
                #check_weibo = sess.run(model.embedded_chars_expanded, batch.feed())
                #check_weibo_shape = sess.run(model.weibo_shape, batch.feed())
                print('train: Epoch {:1d} Batch {:2d} error {:3.1f}%'.format(epoch + 1, i + 1, 100 * train_error))
                print('train: Loss {:3.1f}'.format(train_cost))
                #print check_weibo
                print train_pred
                #print check_weibo_shape


        error = sess.run(model.error, verify.feed())
        print('VERIFY: Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))

    saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
    print 'Finish training. On test set:'

    error = sess.run(model.error, test.feed())
    print('TEST: Error {:2.3f}%'.format(100 * error))

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

        detection_result = sess.run(model.prediction, test_batch.feed())
        prediction.extend(detection_result.tolist())

    # early = sum(test_er)/len(test_er)
    # print('TEST: Early {:2.3f}%'.format(100 * early))
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