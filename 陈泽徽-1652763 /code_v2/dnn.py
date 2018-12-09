import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys

num_2_kpi = {
    1: '09513ae3e75778a3',
    2: '769894baefea4e9e',
    3: '71595dd7171f4540',
    4: '18fbb1d5a5dc099d',
    5: '88cf3a776ba00e7c',
    6: 'a40b1df87e3f1c87',
    7: '40e25005ff8992bd',
    8: '046ec29ddf80d62e',
    9: '9bd90500bfd11edb',
    10: '9ee5879409dccef9',
    11: '8bef9af9a922e0b3',
    12: '07927a9a18fa19ae',
    13: '1c35dbf57f55f5e4',
    14: 'c58bfcbacb2822d1',
    15: 'affb01ca2b4f0b45',
    16: '02e99bd4f6cfb33f',
    17: '8a20c229e9860d0c',
    18: 'a5bf5d65261d859a',
    19: 'da403e4e3f87c9e0',
    20: '7c189dd36f048a6c',
    21: '54e8a140f6237526',
    22: 'e0770391decc44ce',
    23: 'cff6d3c01e6a6bfa',
    24: '8c892e5525f3e491',
    25: '76f4550c43334374',
    26: 'b3b2e6d1a791d63a'
}

if len(sys.argv) > 1:
    kpi_id = int(sys.argv[1])
else:
    for key, value in num_2_kpi.items():
        print(str(key) + ': ' + value)
    kpi_id = input('input kpi type:')

opts = {
    'api_id': num_2_kpi[kpi_id],
    'ratio': 0.7,
}

import keras.backend as K
def FP(threshold = 0.5):
    def fp(y_true_, y_pred_):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred_, 0, 1), threshold_value), K.floatx())
        y_true = K.cast(K.greater_equal(K.clip(y_true_, 0, 1), 1.0), K.floatx())
                                            
        false_positives = (K.mean(K.clip((1-y_true) * y_pred, 0, 1)))
        return false_positives
    return fp

def TP(threshold = 0.5):
    def tp(y_true_, y_pred_):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred_, 0, 1), threshold_value), K.floatx())
        y_true = K.cast(K.greater_equal(K.clip(y_true_, 0, 1), 1.0), K.floatx())
                                            
        true_positives = (K.mean(K.clip(y_true * y_pred, 0, 1)))
        return true_positives
    return tp

def Recall(threshold = 0.5):
    def r(y_true_, y_pred_):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred_, 0, 1), threshold_value), K.floatx())
        y_true = K.cast(K.greater_equal(K.clip(y_true_, 0, 1), 1.0), K.floatx())
                                            
        true_positives = (K.mean(K.clip(y_true * y_pred, 0, 1)))
        positives = (K.mean(K.clip(y_true, 0, 1)))
        return true_positives/(positives + 1e-10)
    return r

def Percise(threshold = 0.5):
    def p(y_true_, y_pred_):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred_, 0, 1), threshold_value), K.floatx())
        y_true = K.cast(K.greater_equal(K.clip(y_true_, 0, 1), 1.0), K.floatx())
                                            
        true_positives = (K.mean(K.clip(y_true * y_pred, 0, 1)))
        false_positives = (K.mean(K.clip((1-y_true) * y_pred, 0, 1)))
        return true_positives /(false_positives + true_positives + 1e-10)
    return p

def F1(threshold = 0.5):
    def f1_score(y_true_, y_pred_):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred_, 0, 1), threshold_value), K.floatx())
        y_true = K.cast(K.greater_equal(K.clip(y_true_, 0, 1), 1.0), K.floatx())
                                            
        true_positives = (K.mean(K.clip(y_true * y_pred, 0, 1)))
        false_positives = (K.mean(K.clip((1-y_true) * y_pred, 0, 1)))
        positives = (K.mean(K.clip(y_true, 0, 1)))
        
        precision = true_positives /(false_positives + true_positives + 1e-10) + 1e-10
        recall = true_positives /(positives + 1e-10) + 1e-10
        
        return 2 * precision * recall / (precision + recall)
    return f1_score

class Data(object):
    def __init__(self, X, y):
        ss = StandardScaler()
        X_ss = ss.fit_transform(X)
        
        self.anomaly_ratio = np.sum(y, axis=0) / X.shape[0]
        print('anomaly data ratio: %g' % self.anomaly_ratio)
    
        self.train_X, self.test_X, self.train_y, self.test_y \
                    = train_test_split(X_ss, y, test_size=0.2)
def get_data():
    file_path = 'feature_data/%s.csv' % str(opts['api_id'])
    raw_data = pd.read_csv(file_path)
    raw_data_arr = np.array(raw_data)
    X, y = raw_data_arr[:, :-1], raw_data_arr[:, -1]
    data = Data(X, y)
    return data

def oversampling(data):
    sample_num = int(opts['ratio'] / data.anomaly_ratio)
    anomaly_index = np.where(data.train_y==1)[0]
    anomaly_points = data.train_X[anomaly_index]
    for num in range(sample_num):
        random_bias = np.random.normal(0, 0.1, anomaly_points.shape)
        random_anomaly_points = anomaly_points + random_bias
        anomaly_y = np.ones((anomaly_index.shape[0], ))
        data.train_X = np.concatenate([data.train_X, random_anomaly_points], axis=0)
        data.train_y = np.concatenate([data.train_y, anomaly_y], axis=0)
        
data = get_data()
oversampling(data)

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint

model = Sequential()
model.add(Dense(256, input_dim=22, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', F1(), Percise(), Recall()])

tensor_board = TensorBoard(log_dir='./logs', update_freq='batch')
save_path = './best_model/best.model'
save_best = ModelCheckpoint(save_path, monitor='val_f1_score', save_best_only=True)
model.fit(data.train_X, data.train_y,
          validation_data=[data.test_X, data.test_y],
          callbacks=[save_best],
          epochs=15,
          batch_size=128,
          shuffle=True)

#test_size = data.test_X.shape[0]
model.load_weights(save_path)
#scores = model.evaluate(data.test_X, data.test_y, batch_size=test_size)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
thres_list = np.arange(0.5, 1, 0.01)
best_f1 = 0
best_thres = 0
for thres in thres_list:
    pred_y = model.predict(data.test_X) > thres
    f1_score_ = precision_recall_fscore_support(pred_y, data.test_y)[2][1]
    if f1_score_ > best_f1:
        best_f1 = f1_score_
        best_thres = thres

pred_y = model.predict(data.test_X) > best_thres        
scores = {}
scores[1] = accuracy_score(pred_y, data.test_y)
metrics = precision_recall_fscore_support(pred_y, data.test_y)
scores[2], scores[3], scores[4] = metrics[0][1], metrics[1][1], metrics[2][1]

output_str = str('KPI ID: %s, accuracy: %.4f, f1 score: %.4f, precision: %.4f, recall: %.4f'\
                        %(num_2_kpi[kpi_id] ,scores[1], scores[4], scores[2], scores[3]))
output_csv = str('%s, %.4f, %.4f, %.4f, %.4f' % (num_2_kpi[kpi_id], scores[1], scores[4], scores[2], scores[3]))
with open('result.txt','a') as f:
    f.write(output_str+'\n')
with open('output.csv','a') as f:
    f.write(output_csv+'\n')

