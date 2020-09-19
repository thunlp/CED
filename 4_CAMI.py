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
train_sz = 5430
verify_sz = 800

test_sz = 1810

epoch_num = 15
batch_sz = 10

K = 200
max_len = 20

vocabulary_size = 20000
num_checkpoints = 5

data_redio = 1.0

data = tf.placeholder(tf.float32, [None, max_len, K])
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


class CAMI():

    def __init__(self, data, target, dropout = 0.1, ws=[7, 5], num_filters=[80, 50], embed_size=200, num_hidden=100, top_k=4, k1=12):

        self.data = data
        self.target = target
        self.data_len = tf.shape(data)[0]
        self.ws = ws
        self.num_filters = num_filters
        self.embed_size = embed_size
        self.num_hidden = num_hidden
        self.top_k = top_k
        self.k1 = k1
        self.dropout_keep_prob = dropout

        self.cur_seed = random.getrandbits(64)
        self.l2_reg_lambda = 0.002

        # Initialize regular item
        self.l2_loss = tf.constant(0.0)

        # embedding
        #w_length = tf.cast(max_len * self.data_len, dtype=tf.int32)
        #sent = tf.reshape(self.data, [w_length, -1])

        #sent = self.data
        #self.W = tf.Variable(tf.random_uniform([vocabulary_size, self.embed_size], -1.0, 1.0), name="embed_W")
        #self.sent_embed = tf.nn.embedding_lookup(self.W, sent)
        #self.input = tf.expand_dims(self.sent_embed, -1)

        self.input = tf.expand_dims(self.data, -1)

        self.W1 = tf.Variable(tf.truncated_normal([self.ws[0], self.embed_size, 1, self.num_filters[0]], stddev=0.1), name="W1")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[self.num_filters[0]]), "b1")

        self.W2 = tf.Variable(tf.truncated_normal([self.ws[1], self.num_filters[0], 1, self.num_filters[1]], stddev=0.1), name="W2")
        self.b2 = tf.Variable(tf.constant(0.1, shape=[self.num_filters[1]]), "b2")

        self.Wh = tf.Variable(tf.truncated_normal([self.top_k*self.num_filters[1], self.num_hidden], stddev=0.1), name= "Wh")
        self.bh = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]), name="bh")

        self.Wo = tf.Variable(tf.truncated_normal([self.num_hidden, 2], stddev=0.01), name="Wo")

        self.weight = tf.get_variable(name="weight", shape=[self.top_k*self.num_filters[1], 2],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.cur_seed))
        self.bias = tf.get_variable(name="bias", shape=[2],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.cur_seed))

        self.l2_loss += tf.nn.l2_loss(self.weight)
        self.l2_loss += tf.nn.l2_loss(self.bias)

        self.CAMI_output
        self.error
        self.cnn_optimize

    @lazy_property
    def per_dim_conv_layer1(self):
        x = self.input
        w = self.W1
        b = self.b1
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu1")
        #input_unstack = tf.cast(x, tf.float32)
        #input_unstack = tf.unstack(input_unstack, axis=2)
        #w_unstack = tf.unstack(w, axis=1)
        #b_unstack = tf.unstack(b, axis=1)
        #convs = []
        #with tf.name_scope("per_dim_conv"):
        #    for i in range(len(input_unstack)):
        #        conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="VALID") + b_unstack[i])#[batch_size, k1+ws2-1, num_filters[1]]
        #        convs.append(conv)
        #    conv = tf.stack(convs, axis=2)
            #[batch_size, k1+ws-1, embed_size, num_filters[1]]
        return h

    @lazy_property

    def fold_k_max_pooling1(self):
        x = self.per_dim_conv_layer1
        k = self.k1
        pooled = tf.nn.max_pool(x, ksize=[1, max_len - self.ws[0] + 1 - k +1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool1")
        #input_unstack = tf.unstack(x, axis=2)
        #out = []
        #with tf.name_scope("fold_k_max_pooling"):
        #    for i in range(0, len(input_unstack), 2):
        #        fold = tf.add(input_unstack[i], input_unstack[i+1])#[batch_size, k1, num_filters[1]]
        #        conv = tf.transpose(fold, perm=[0, 2, 1])
        #        values = tf.nn.top_k(conv, k, sorted=False).values #[batch_size, num_filters[1], top_k]
        #        values = tf.transpose(values, perm=[0, 2, 1])
        #        out.append(values)
        #    fold = tf.stack(out, axis=2)#[batch_size, k2, embed_size/2, num_filters[1]]
        pooled = tf.reshape(pooled, [-1, k, self.num_filters[0], 1])
        return pooled

    @lazy_property
    def per_dim_conv_layer2(self):
        x = self.fold_k_max_pooling1
        w = self.W2
        b = self.b2
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu1")
        #input_unstack = tf.unstack(x, axis=2)
        #w_unstack = tf.unstack(w, axis=1)
        #b_unstack = tf.unstack(b, axis=1)
        #convs = []
        #with tf.name_scope("per_dim_conv"):
        #    for i in range(len(input_unstack)):
        #        conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="VALID") + b_unstack[i])#[batch_size, k1+ws2-1, num_filters[1]]
        #        convs.append(conv)
        #    conv = tf.stack(convs, axis=2)
        #    #[batch_size, k1+ws-1, embed_size, num_filters[1]]
        return h

    @lazy_property
    def fold_k_max_pooling2(self):
        x = self.per_dim_conv_layer2
        k = self.top_k
        pooled = tf.nn.max_pool(x, ksize=[1, self.k1 - self.ws[1] + 1 - k + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool1")
        #input_unstack = tf.unstack(x, axis=2)
        #out = []
        #with tf.name_scope("fold_k_max_pooling"):
        #    for i in range(0, len(input_unstack), 2):
        #        fold = tf.add(input_unstack[i], input_unstack[i+1])#[batch_size, k1, num_filters[1]]
        #        conv = tf.transpose(fold, perm=[0, 2, 1])
        #        values = tf.nn.top_k(conv, k, sorted=False).values #[batch_size, num_filters[1], top_k]
        #        values = tf.transpose(values, perm=[0, 2, 1])
        #        out.append(values)
        #    fold = tf.stack(out, axis=2)#[batch_size, k2, embed_size/2, num_filters[1]]
        return pooled

    # @lazy_property
    # def full_connect_layer(self):
    #     x = self.fold_flatten
    #     w = self.Wh
    #     b = self.bh
    #     wo = self.Wo
    #     weight = self.weight
    #     bias = self.bias
    #     dropout_keep_prob = self.dropout_keep_prob
    #     with tf.name_scope("full_connect_layer"):
    #         h = tf.nn.tanh(tf.matmul(x, w) + b)
    #         h = tf.nn.dropout(h, dropout_keep_prob)
    #         o = tf.matmul(h, wo)
    #         o = tf.reshape(o, [self.data_len, -1])
    #         out = tf.nn.xw_plus_b(o, weight, bias)
    #     return out

    @lazy_property
    def CAMI_output(self):
        x = tf.reshape(self.fold_k_max_pooling2, [-1, self.top_k*self.num_filters[1]])
        w = self.Wh
        b = self.bh
        wo = self.Wo
        weight = self.weight
        bias = self.bias
        dropout_keep_prob = self.dropout_keep_prob
        #with tf.name_scope("full_connect_layer"):
        #    h = tf.nn.tanh(tf.matmul(x, w) + b)
        #    h = tf.nn.dropout(h, dropout_keep_prob)
        #    #out = tf.matmul(h, wo)
        #    out = tf.nn.xw_plus_b(h, weight, bias)

        out = tf.nn.tanh(tf.matmul(x, w) + b)
        out = tf.nn.dropout(out, dropout_keep_prob)
        out = tf.matmul(out, wo)
        #out = tf.nn.bias_add(out, bias)
        #out = tf.nn.xw_plus_b(x, weight, bias)
        #out = tf.nn.dropout(out, dropout_keep_prob)
        return out

    @lazy_property
    def CAMI_output_softmax(self):
        return tf.nn.softmax(self.CAMI_output)

    # @lazy_property
    # def CAMI_output(self):
    #     self.fold_flatten = tf.reshape(self.fold_k_max_pooling2, [-1, self.top_k*10])
    #     out = self.full_connect_layer
    #     #out = tf.nn.softmax(out)
    #     return out

    @lazy_property
    def detection_result(self):
        result = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.CAMI_output_softmax, 1))
        return result

    @lazy_property
    def cnn_cost(self):  # averagely2)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.CAMI_output, labels=self.target)
        return tf.reduce_sum(cross_entropy) + self.l2_reg_lambda * self.l2_loss

    @lazy_property
    def cnn_optimize(self):
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cnn_cost, global_step=global_step)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.CAMI_output_softmax, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

class DataTensor:
    def __init__(self, namelist):
        length = len(namelist)
        self.tensor = np.zeros((length, 2))
        self.data = np.zeros((length, max_len, K))

        for j in range(length):
            name = namelist[j]
            psg = np.array(seq_vec_dic[name])
            len_psg = len(psg)
            len_psg = int(len_psg * data_redio)
            self.data[j, :len_psg, :] = psg[:len_psg, :]
            #self.len[j] = class_dic[name]['len']
            self.tensor[j, :] = class_dic[name]['class']
            # if len(psg) > max_len:
                # temp = psg[:max_len, :]
                # #print temp.shape
                # self.data[j, :max_len, :] = temp
            # else:
                # #print psg.shape
                # self.data[j, :len(psg), :] = psg

    def feed(self):
        return {data: self.data, target: self.tensor}

if __name__ == '__main__':

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
    print 'BROKEN:', len(broken)
    for name in broken:
        seq_vec_dic.pop(name)
    print 'broken handled'
    print len(seq_vec_dic)

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

    model = CAMI(data, target)

    sess = tf.Session()

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "4_CAMI_runs", timestamp))
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

    print 'DataTensor Done'

    max_accu = 0
    best_at_step = 0

    for epoch in range(epoch_num):
        for i in range(train_sz/batch_sz):
            batch = DataTensor(train_names[i*batch_sz:(i+1)*batch_sz])
            sess.run(model.cnn_optimize, batch.feed())   # learning here
            current_step = tf.train.global_step(sess, global_step)
            #print current_step

            # CHECK
            #check_data = sess.run(model.data, batch.feed())
            #print 'data:', check_data
            #check_per_dim_conv_layer1 = sess.run(model.per_dim_conv_layer1, batch.feed())
            #print check_per_dim_conv_layer1
            #check_fold_k_max_pooling1 = sess.run(model.fold_k_max_pooling1, batch.feed())
            #print check_fold_k_max_pooling1
            #check_per_dim_conv_layer2 = sess.run(model.per_dim_conv_layer2, batch.feed())
            #print check_per_dim_conv_layer2
            #check_fold_k_max_pooling2 = sess.run(model.fold_k_max_pooling2, batch.feed())
            #print check_fold_k_max_pooling2
            #check_CAMI_output = sess.run(model.CAMI_output, batch.feed())
            #print check_CAMI_output

            if current_step % 100 == 0:
                prediction = []
                for i in range(verify_sz/batch_sz):
                    verify_batch = DataTensor(verify_names[i*batch_sz:(i+1)*batch_sz])
                    detection_result = sess.run(model.CAMI_output_softmax, verify_batch.feed())
                    prediction.extend(detection_result.tolist())
                verify_pred = np.argmin(prediction, 1)
                np.array(verify_pred)
                verify_accu = accuracy_score(verify_true, verify_pred)
                if verify_accu >= max_accu:
                    max_accu = verify_accu
                    best_at_step = current_step
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print 'Best of valid = {}, at step {}'.format(max_accu, best_at_step)

            if i % 10 ==0:
                train_error = sess.run(model.error, batch.feed())
                train_cost = sess.run(model.cnn_cost,batch.feed())
                train_pred = sess.run(model.CAMI_output_softmax,batch.feed())
                print('train: Epoch {:1d} Batch {:2d} error {:3.1f}%'.format(epoch + 1, i + 1, 100 * train_error))
                print('train: Loss {:3.1f}'.format(train_cost))
                print train_pred

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

            detection_result = sess.run(model.CAMI_output_softmax, train_batch.feed())
            train_prediction.extend(detection_result.tolist())
        # early = sum(train_er)/len(train_er)
        # print('TRAIN: Early {:2.3f}%'.format(100 * early))
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

            detection_result = sess.run(model.CAMI_output_softmax, verify_batch.feed())
            verify_prediction.extend(detection_result.tolist())
        # early = sum(verify_er)/len(verify_er)
        # print('VERIFY: Early {:2.3f}%'.format(100 * early))
        verify_accuracy = float(verify_right_one)/verify_sz
        print('VERIFY: Error {:2.3f}%'.format(100 * verify_accuracy))
        verify_pred = np.argmin(verify_prediction, 1)
        np.array(verify_pred)
        Accu = accuracy_score(verify_true, verify_pred)
        print('VERIFY: Accuracy {:2.3f}%'.format(100 * Accu))

    saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
    print 'Finish training. On test set:'
    print 'Best step:', best_at_step

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

        detection_result = sess.run(model.CAMI_output_softmax, test_batch.feed())
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