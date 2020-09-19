from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import codecs
import json
import math
import random

from sklearn.svm import SVC

# hyper-params
train_sz = 6030
test_sz = 2010
K = 1000

data_redio = 1.0

def get_tensors(namelist):
    length = len(namelist)
    data_tensor = []
    class_tensor = []
    y_true = []
    for j in range(length):
        name = namelist[j]
        psg = np.array(seq_vec_dic[name])
        self_length = len(seq_vec_dic[name])
        self_length = int(self_length * data_redio)
        psg_temp = [0] * K
        for i in range(self_length):
            psg_temp = map(lambda (a, b): a + b, zip(psg_temp, psg[i]))
        data_tensor.append(psg_temp)
        class_tensor.append(np.argmax(class_dic[name]['class']))
        y_true.append(1 - np.argmax(class_dic[name]['class']))
    return data_tensor, class_tensor, y_true


if __name__ == '__main__':
    # We treat images as sequences of pixel rows.
    with codecs.open("10_parted_posts_seqvec.txt", "r", 'utf-8') as f:
        seq_vec_dic = json.load(f, encoding='utf-8')
    with codecs.open("class_8050.json", "r", 'utf-8') as f:
        class_dic = json.load(f, encoding='utf-8')
    print 'file read'

    # # handle all zero vectors
    # broken = []
    # for name, psg in seq_vec_dic.iteritems():
        # if psg == []:
            # broken.append(name)
    # for name in broken:
        # seq_vec_dic.pop(name)
    # print 'broken handled'

    keys = seq_vec_dic.keys()
    random.shuffle(keys)

    train_names = keys[0: train_sz]
    test_names = keys[train_sz:train_sz + test_sz]
    print 'train_num:', len(train_names)
    print 'test_num:', len(test_names)

    test_data_tensor, test_class_tensor, test_true = get_tensors(test_names)
    train_data_tensor, train_class_tensor, train_true = get_tensors(train_names)

    np.array(test_data_tensor)
    np.array(test_class_tensor)
    np.array(test_true)
    np.array(train_data_tensor)
    np.array(train_class_tensor)
    np.array(train_true)

    clf_all = SVC()
    clf_all.fit(train_data_tensor, train_class_tensor)
    prediction = clf_all.predict(test_data_tensor)
    test_pred = [1 - x for x in prediction]

    p = precision_score(test_true, test_pred, average='binary')
    r = recall_score(test_true, test_pred, average='binary')
    f1score = f1_score(test_true, test_pred, average='binary')
    accu = clf_all .score(test_data_tensor, test_class_tensor)
    print 'P', p
    print 'R', r
    print 'F1', f1score
    print 'Accuracy:', accu



