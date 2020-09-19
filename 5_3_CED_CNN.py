from __future__ import print_function

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
K = 100

vocabulary_size = 20000
embedding_size = 200
weibo_len = 30

num_checkpoints = 5

# the threshold value
alpha = 0.95
# hyper-params in AttenCED loss
lambd0 = 0.01
lambd1 = 0.2

data_redio = 1.0


weibo = tf.placeholder(tf.int32, [None, weibo_len])
data_id = tf.placeholder(tf.float32, [None, max_len, K])
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

    def __init__(self, weibo, data_id, data_len, target, num_hidden=100, num_layers=2, weibo_len=weibo_len,
                 num_filters=50, filter_sizes=(4, 5)):

        self.weibo = weibo
        self.data_id = data_id
        self.data_len = data_len
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.num_classes = int(self.target.get_shape()[1])
        self.l2_reg_lambda = 0.001
        self.cur_seed = random.getrandbits(32)

        # Add regular item
        self.l2_loss = tf.constant(0.0)

        # CNN
        self.W = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),name="W")
        # Embedding
        self.weibo_embedded_chars = tf.nn.embedding_lookup(self.W, self.weibo)
        self.weibo_embedded_chars_expanded = tf.expand_dims(self.weibo_embedded_chars, -1)
        self.weibo_embedded_shape = tf.shape(self.weibo_embedded_chars_expanded)

        self.data_id = tf.cast(self.data_id, tf.int32)
        self.data_embedded_chars = tf.nn.embedding_lookup(self.W, self.data_id)
        self.data_embedded_chars = tf.reshape(self.data_embedded_chars, [-1, K, embedding_size])
        self.data_embedded_chars_expanded = tf.expand_dims(self.data_embedded_chars, -1)
        self.data_embedded_shape = tf.shape(self.data_embedded_chars_expanded)

        # Msg create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, mean=1.0), name="W")
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.weibo_embedded_chars_expanded,
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


        # Post create a convolution + maxpool layer for each filter size
        post_pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, mean=1.0), name="W")
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.data_embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, K - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                post_pooled_outputs.append(pooled)
        # combine all the pooled features
        self.post_num_filters_total = num_filters * len(filter_sizes)
        self.post_h_pool = tf.concat(post_pooled_outputs, 3)
        self.data = tf.reshape(self.post_h_pool, [-1, max_len, self.post_num_filters_total])


        # rnn
        self.output, self.state = tf.nn.dynamic_rnn(
            tf.contrib.rnn.GRUCell(self._num_hidden),
            self.data,
            dtype=tf.float32,
            sequence_length=self.data_len,
        )


        self.weight = tf.get_variable(name="weight", shape=[self._num_hidden + self.num_filters_total, self.num_classes],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.cur_seed))
        self.bias = tf.get_variable(name="bias", shape=[self.num_classes],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.cur_seed))
        self.M = tf.get_variable(name="M", shape=[self.num_filters_total, self._num_hidden],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.cur_seed))

        self.l2_loss += tf.nn.l2_loss(self.M)
        self.l2_loss += tf.nn.l2_loss(self.weight)
        self.l2_loss += tf.nn.l2_loss(self.bias)

        self.error
        self.cnn_optimize
        self.cnn_detection
        self.train_loss

    @lazy_property
    def length(self):
        length = self.data_len
        length = tf.cast(length, tf.float32)
        # redio = tf.constant(data_redio, dtype=tf.float32, shape=[1, batch_sz])
        # redio = tf.constant(data_redio, dtype=tf.float32, shape=[1, batch_sz])
        length = tf.floor(tf.multiply(length, data_redio))
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def train_loss(self):

        bias = self.bias
        weight = self.weight
        output = self.output
        h_pool_flat = self.h_pool_flat

        # output = tf.div(output, 10)
        # h_pool_flat = tf.div(h_pool_flat, 10)

        length = self.data_len
        M = self.M
        batch_size = tf.shape(output)[0]
        cnn_output_size = tf.shape(h_pool_flat)[1]
        output_size = int(output.get_shape()[2])

        # Get label y
        label = tf.argmax(self.target, 1)
        the_ones = tf.cast(tf.ones([batch_size]), tf.int64)
        label = tf.subtract(the_ones, label)

        loss0 = tf.zeros([0, 1])
        earl0 = tf.zeros([0, 1])
        i0 = tf.constant(0)
        def while_condition_outer(loss, early, i):
            return i < batch_size
        def body_outer(loss, early, i):

            # Determine the attentioned first step meeting threshold
            # Calculate the attentioned hh
            j0 = tf.constant(0)
            first_place0 = tf.constant(-1)
            while_condition_inter1 = lambda first_place, j: j < tf.minimum(length[i], max_len)
            def body_inter1(first_place, j):
                output_temp = tf.reshape(output[i][j], [1, output_size])
                h_pool_flat_i = tf.reshape(h_pool_flat[i], [1, cnn_output_size])
                pre_temp = tf.concat([h_pool_flat_i, output_temp], 1)
                pre_temp = tf.reshape(pre_temp, [1, cnn_output_size + output_size])
                pre_temp = tf.nn.xw_plus_b(pre_temp, weight, bias)
                pre_temp = tf.nn.softmax(pre_temp)

                first_place_dependened = tf.cond(tf.equal(first_place, -1), lambda: True, lambda: False)
                now_place = tf.cond(tf.abs(pre_temp[0][0] - pre_temp[0][1]) > alpha, lambda: j, lambda: -1)
                first_place = tf.cond(first_place_dependened, lambda: now_place, lambda: first_place)

                last_place_dependened = tf.cond(tf.equal(j, tf.minimum(length[i], max_len) - 1), lambda: True, lambda: False)
                last_place_dependened = tf.cond(tf.equal(first_place, -1), lambda: last_place_dependened, lambda: False)
                first_place = tf.cond(last_place_dependened, lambda: tf.minimum(length[i], max_len) - 1, lambda: first_place)

                return (first_place, tf.add(j, 1))
            the_first_place, _ = tf.while_loop(while_condition_inter1, body_inter1, [first_place0, j0],
                                                                shape_invariants=[first_place0.get_shape(), j0.get_shape()])


            # Calculate loss
            r0 = the_first_place
            loss_temp0 = tf.zeros([1, 1])
            loss_temp0 = tf.cast(loss_temp0, tf.float32)
            while_condition_inter2 = lambda loss_temp, r: r < tf.minimum(length[i], max_len)
            def body_inter2(loss_temp, r):

                output_temp = tf.reshape(output[i][r], [1, output_size])
                h_pool_flat_i = tf.reshape(h_pool_flat[i], [1, cnn_output_size])
                pre_temp = tf.concat([h_pool_flat_i, output_temp], 1)
                pre_temp = tf.reshape(pre_temp, [1, cnn_output_size + output_size])
                pre_temp = tf.nn.xw_plus_b(pre_temp, weight, bias)
                target_temp = target[i]
                log_likelihood = tf.nn.softmax_cross_entropy_with_logits(logits=pre_temp, labels=target_temp)
                log_likelihood = tf.reshape(log_likelihood, [1, 1])

                lambd0_now = tf.reshape(lambd0, [1, 1])
                label_now = tf.cast(label[i], tf.float32)
                label_now = tf.reshape(label_now, [1, 1])

                # Calculate (O-prediction + lambda0 * O-diff)
                x1 = tf.nn.relu(tf.subtract(tf.log(alpha), log_likelihood))
                x2 = tf.nn.relu(tf.subtract(log_likelihood, tf.log(1 - alpha)))
                y = tf.add(tf.multiply(label_now, x1), tf.multiply(tf.subtract(tf.ones([1, 1]), label_now), x2))
                y = tf.multiply(lambd0_now, y)
                loss_temp_now = tf.subtract(log_likelihood, y)
                loss_temp = tf.add(loss_temp, loss_temp_now)

                return (loss_temp, tf.add(r, 1))
            loss_for_one, _ = tf.while_loop(while_condition_inter2, body_inter2, [loss_temp0, r0], shape_invariants=[tf.TensorShape([1, 1]), i0.get_shape()])

            length_still = tf.reshape(length[i] - the_first_place, [1, 1])
            length_still = tf.cast(length_still, tf.float32)
            loss_for_one = tf.div(loss_for_one, length_still)

            # Calculate O-time
            lambd1_now = tf.reshape(lambd1, [1, 1])
            the_first_place_now = tf.reshape(the_first_place + 1, [1, 1])
            the_first_place_now = tf.cast(the_first_place_now, tf.float32)
            length_now = tf.reshape(length[i], [1, 1])
            length_now = tf.cast(length_now, tf.float32)
            loss_now = tf.div(the_first_place_now, length_now)
            loss_now = tf.log(loss_now)
            loss_now = tf.multiply(lambd1_now, loss_now)

            # Calculate CED loss
            loss_for_one = tf.subtract(loss_for_one, loss_now)

            loss = tf.concat([loss, loss_for_one], 0)

            # the_first_place is the first step length meeting threshold
            the_first_place = tf.cast(the_first_place + 1, tf.float32)
            the_first_place = tf.reshape(the_first_place, [1, 1])
            early = tf.concat([early, the_first_place], 0)

            return [loss, early, tf.add(i, 1)]
        loss, first_to_detetmin, _ = tf.while_loop(while_condition_outer, body_outer,[loss0, earl0, i0],
                        shape_invariants=[tf.TensorShape([None, 1]), tf.TensorShape([None, 1]), i0.get_shape()])

        self.first_to_detetmin = tf.cast(first_to_detetmin, tf.float32)
        return loss

    @lazy_property
    def attentioned_prediction_output(self):
        output = self.output
        bias = self.bias
        weight = self.weight
        h_pool_flat = self.h_pool_flat

        #output = tf.div(output, 10)
        #h_pool_flat = tf.div(h_pool_flat, 10)

        length = self.length
        M = self.M
        batch_size = tf.shape(output)[0]
        cnn_output_size = tf.shape(h_pool_flat)[1]
        output_size = int(output.get_shape()[2])

        h0 = tf.zeros([0, output_size])
        earl0 = tf.zeros([0, 1])
        i0 = tf.constant(0)
        tocheck0 = tf.ones([1, 1])
        # tocheck0 = tf.ones([1, output_size])

        def while_condition_outer(tocheck, h, early, i):
            return i < batch_size
        def body_outer(tocheck, h, early, i):

            # Determine the attentioned first step meeting threshold
            # Calculate the attentioned hh
            j0 = tf.constant(0)
            first_place0 = tf.constant(-1)
            while_condition_inter1 = lambda first_place, j: j < tf.minimum(length[i], max_len)
            def body_inter1(first_place, j):
                output_temp = tf.reshape(output[i][j], [1, output_size])
                h_pool_flat_i = tf.reshape(h_pool_flat[i], [1, cnn_output_size])
                pre_temp = tf.concat([h_pool_flat_i, output_temp], 1)
                pre_temp = tf.reshape(pre_temp, [1, cnn_output_size + output_size])
                pre_temp = tf.nn.xw_plus_b(pre_temp, weight, bias)
                pre_temp = tf.nn.softmax(pre_temp)

                first_place_dependened = tf.cond(tf.equal(first_place, -1), lambda: True, lambda: False)
                now_place = tf.cond(tf.abs(pre_temp[0][0] - pre_temp[0][1]) > alpha, lambda: j, lambda: -1)
                first_place = tf.cond(first_place_dependened, lambda: now_place, lambda: first_place)

                last_place_dependened = tf.cond(tf.equal(j, tf.minimum(length[i], max_len) - 1), lambda: True, lambda: False)
                last_place_dependened = tf.cond(tf.equal(first_place, -1), lambda: last_place_dependened, lambda: False)
                first_place = tf.cond(last_place_dependened, lambda: tf.minimum(length[i], max_len) - 1, lambda: first_place)

                return (first_place, tf.add(j, 1))
            the_first_place, _ = tf.while_loop(while_condition_inter1, body_inter1, [first_place0, j0],
                                                                shape_invariants=[first_place0.get_shape(), j0.get_shape()])


            #tocheck = atten_check

            h_temp = tf.reshape(output[i][the_first_place], [1, output_size])
            h = tf.concat([h, h_temp], 0)

            the_first_place = tf.cast(the_first_place + 1, tf.float32)
            the_first_place = tf.reshape(the_first_place, [1, 1])
            early = tf.concat([early, the_first_place], 0)
            #h = tf.concat(0, [h, h_atten])

            return [tocheck, h, early, tf.add(i, 1)]
        check, h, first_to_detetmin, i_n = tf.while_loop(while_condition_outer, body_outer, [tocheck0, h0, earl0, i0],
                             shape_invariants=[tf.TensorShape([1, None]), tf.TensorShape([None, output_size]), tf.TensorShape([None, 1]), i0.get_shape()])
        self.shape = tf.shape(check)
        self.prediction_parameter = check

        self.first_to_detetmin = tf.cast(first_to_detetmin, tf.float32)

        h_new = tf.concat([h_pool_flat, h], 1)
        
        return h_new

    @lazy_property
    def cnn_cost(self):
        #return -tf.reduce_sum(self.target * tf.log(self.cnn_prediction)) + self.l2_reg_lambda * self.l2_loss
        return tf.reduce_sum(self.train_loss) + self.l2_reg_lambda * self.l2_loss

    @lazy_property
    def cnn_optimize(self):
        learning_rate = 0.001
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cnn_cost, global_step=global_step)

    @lazy_property
    def cnn_detection(self):
        h = self.attentioned_prediction_output
        prediction = tf.nn.xw_plus_b(h, self.weight, self.bias)
        # prediction = tf.nn.sigmoid(prediction)
        prediction = tf.nn.softmax(prediction)
        return prediction

    @lazy_property
    def detection_result(self):
        result = tf.equal(
            tf.argmax(self.target, 1), tf.argmax(self.cnn_detection, 1))
        return result

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.cnn_detection, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @lazy_property
    def early(self):
        length = tf.cast(self.data_len, tf.float32)
        first_to_detetmin = tf.transpose(self.first_to_detetmin)
        # calculate Early Rate
        early_position = tf.div(first_to_detetmin, length)
        return tf.reduce_mean(early_position)


class DataTensor:
    def __init__(self, namelist):
        length = len(namelist)
        self.weibo = np.zeros((length, weibo_len))
        self.len = np.zeros(length, dtype=np.int32)
        self.tensor = np.zeros((length, 2))
        self.data = np.zeros((length, max_len, K))
        for j in range(length):
            name = namelist[j]
            psg = np.array(seq_vec_dic[name])
            #self.len[j] = class_dic[name]['len']
            self.len[j] = len(seq_vec_dic[name])
            self.tensor[j, :] = class_dic[name]['class']
            self.weibo[j, :] = msg_seq_vec_dic[name]
            self.data[j, :len(psg), :] = psg
            # if len(psg) > max_len:
                # temp = psg[:max_len, :]
                # #print temp.shape
                # self.data[j, :max_len, :] = temp
            # else:
                # #print psg.shape
                # self.data[j, :len(psg), :] = psg

    def feed(self):
        return {weibo: self.weibo, data_id: self.data, data_len: self.len, target: self.tensor}

target
if __name__ == '__main__':

    with codecs.open("post_id.json", "r", 'utf-8') as f:
        seq_vec_dic = json.load(f, encoding='utf-8')
    with codecs.open("msg_id.json", "r", 'utf-8') as f:
        msg_seq_vec_dic = json.load(f, encoding='utf-8')
    with codecs.open("class_8050.json", "r", 'utf-8') as f:
        class_dic = json.load(f, encoding='utf-8')
    print ('file read')

    # handle all zero vectors
    broken = []
    for name, psg in seq_vec_dic.iteritems():
        #if class_dic[name]['len'] < 20 or psg == []:
        if psg == []:
            broken.append(name)
    for name in broken:
        seq_vec_dic.pop(name)
    print ('broken handled')

    keys = seq_vec_dic.keys()
    
    random.seed(32)
    random.shuffle(keys)

    train_names = keys[0:train_sz]
    verify_names = keys[train_sz:train_sz+verify_sz]
    test_names = keys[train_sz+verify_sz:train_sz + verify_sz + test_sz]
    print ('train_num:', len(train_names))
    print ('verify_num', len(verify_names))
    print ('test_num:', len(test_names))
    # exit(0)

    model = VariableSequenceClassification(weibo, data_id, data_len, target)

    sess = tf.Session()

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "5_3_CED_CNN_runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

    # sess.graph.finalize()

    sess.run(tf.global_variables_initializer())

    test = DataTensor(test_names)
    verify = DataTensor(verify_names)
    train = DataTensor(train_names)

    test_true = np.argmin(test.tensor, 1)
    np.array(test_true)
    print (test_true)
    train_true = np.argmin(train.tensor, 1)
    np.array(train_true)
    verify_true = np.argmin(verify.tensor, 1)
    np.array(verify_true)

    print ('Test and Train DataTensor Done')

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
                    print ('Best of valid = {}, at step {}'.format(max_accu, best_at_step))
                
                verify_er = []
                for i in range(verify_sz/batch_sz):
                    verify_batch = DataTensor(verify_names[i*batch_sz:(i+1)*batch_sz])
                    early_result = sess.run(model.early, verify_batch.feed())
                    early_result = early_result.tolist()
                    if not math.isnan(early_result):
                        verify_er.append(early_result)
                early = sum(verify_er)/len(verify_er)
                print('VERIFY Curve: Early {:2.3f}% at step {}'.format(100 * early, current_step))

            if i % 10 == 0:
                train_error = sess.run(model.error, batch.feed())
                train_cost = sess.run(model.cnn_cost, batch.feed())
                train_pred = sess.run(model.cnn_detection, batch.feed())
                #check_weibo = sess.run(model.embedded_chars_expanded, batch.feed())
                #check_weibo_shape = sess.run(model.weibo_shape, batch.feed())
                print('train: Epoch {:1d} Batch {:2d} error {:3.1f}%'.format(epoch + 1, i + 1, 100 * train_error))
                print('train: Loss {:3.1f}'.format(train_cost))
                #print check_weibo
                print (train_pred)
                #print check_weibo_shape

        # Verify into several steps: True means right prediction
        # error = sess.run(model.error, verify.feed())
        # print('VERIFY: Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
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

            early_result = sess.run(model.early, train_batch.feed())
            early_result = early_result.tolist()
            if not math.isnan(early_result):
                train_er.append(early_result)

            detection_result = sess.run(model.detection_result, train_batch.feed())
            detection_result = detection_result.tolist()
            true_num = detection_result.count(True)
            train_right_one = train_right_one + true_num

            detection_result = sess.run(model.cnn_detection, train_batch.feed())
            train_prediction.extend(detection_result.tolist())
        early = sum(train_er)/len(train_er)
        print('TRAIN: Early {:2.3f}%'.format(100 * early))
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

            early_result = sess.run(model.early, verify_batch.feed())
            early_result = early_result.tolist()
            if not math.isnan(early_result):
                verify_er.append(early_result)

            detection_result = sess.run(model.detection_result, verify_batch.feed())
            detection_result = detection_result.tolist()
            true_num = detection_result.count(True)
            verify_right_one = verify_right_one + true_num

            detection_result = sess.run(model.cnn_detection, verify_batch.feed())
            verify_prediction.extend(detection_result.tolist())
        early = sum(verify_er)/len(verify_er)
        print('VERIFY: Early {:2.3f}%'.format(100 * early))
        verify_accuracy = float(verify_right_one)/verify_sz
        print('VERIFY: Error {:2.3f}%'.format(100 * verify_accuracy))
        verify_pred = np.argmin(verify_prediction, 1)
        np.array(verify_pred)
        Accu = accuracy_score(verify_true, verify_pred)
        print('VERIFY: Accuracy {:2.3f}%'.format(100 * Accu))

    saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
    print ('Finish training. On test set:')

    test_er = []
    right_one = 0
    prediction = []
    for i in range(test_sz/batch_sz):
        test_batch = DataTensor(test_names[i*batch_sz:(i+1)*batch_sz])

        early_result = sess.run(model.early, test_batch.feed())
        early_result = early_result.tolist()
        if not math.isnan(early_result):
            test_er.append(early_result)

        detection_result = sess.run(model.detection_result, test_batch.feed())
        detection_result = detection_result.tolist()
        true_num = detection_result.count(True)
        right_one = right_one + true_num

        detection_result = sess.run(model.cnn_detection, test_batch.feed())
        prediction.extend(detection_result.tolist())

    early = sum(test_er)/len(test_er)
    print('TEST: Early {:2.3f}%'.format(100 * early))
    test_accuracy = float(right_one)/test_sz
    print('TEST: Error {:2.3f}%'.format(100 * test_accuracy))
    print (prediction)

    test_pred = np.argmin(prediction, 1)
    np.array(test_pred)
    print (test_pred)
    P = precision_score(test_true, test_pred)
    R = recall_score(test_true, test_pred)
    Accu = accuracy_score(test_true, test_pred)
    f1score = f1_score(test_true, test_pred, average='binary')
    print ('Accu', Accu)
    print ('P', P)
    print ('R', R)
    print ('F1', f1score)