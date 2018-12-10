# coding: utf-8
import pandas as pd
import numpy as np

import keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import objectives
import keras.backend as K
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sys
plt.style.use('ggplot')

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
	kpi_id = 16

kpi_id = int(kpi_id)
trainData = pd.read_csv('feature_data/' + str(num_2_kpi[kpi_id]) + '.csv')

# here is for feature engineering data process
data = np.array(trainData)
data = preprocessing.minmax_scale(data)

trainX = np.array(data[...,:-1])
trainY = np.array(data[..., -1])
trainX = keras.utils.normalize(trainX)

x_train, x_valid, y_train, y_valid = train_test_split(trainX, trainY, test_size=0.2, shuffle=False)

batch_size = 100

data_size = x_train.shape[0]
anomaly_ratio = 0.5
enlarge_data_size = int(data_size * anomaly_ratio)

x_anomaly_train = x_train[y_train==1]
x_anomaly_valid = x_valid[y_valid==1]

train_reshape_size = int(x_anomaly_train.shape[0] / batch_size) * batch_size
valid_reshape_size = int(x_anomaly_valid.shape[0] / batch_size) * batch_size

x_anomaly_train = x_anomaly_train[:train_reshape_size]
x_anomaly_valid = x_anomaly_valid[:valid_reshape_size]

#print(len(x_anomaly_train), len(x_anomaly_valid))

def VAE_models(original_dim = 14, latent_dim = 2, intermediate_dim = 7):
    epsilon_std = 1.0

    x = Input(shape=(original_dim, ))
    h = Dense(intermediate_dim, activation='relu')(x)
    h2 = Dense(intermediate_dim, activation='relu')(h)
    z_mean = Dense(latent_dim)(h2)
    z_log_var = Dense(latent_dim)(h2)

    #my tips:Gauss sampling,sample Z
    def sampling(args): 
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # my tips: get sample z(encoded)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_h2 = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    
    h_decoded = decoder_h(z)
    h_decoded2 = decoder_h2(h_decoded)
    x_decoded_mean = decoder_mean(h_decoded2)
    
    z_ = Input(shape=(latent_dim, ))
    h_decoded_ = decoder_h(z_)
    h_decoded2_ = decoder_h2(h_decoded_)
    x_decoded_mean_ = decoder_mean(h_decoded2_)

    #my tips:loss(restruct X)+KL
    def vae_loss(x, x_decoded_mean):
          #my tips:logloss
        xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        #my tips:see paper's appendix B
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae = Model(x, x_decoded_mean)
    encoder = Model(x, z_mean)
    decoder = Model(z_, x_decoded_mean_)
    
    vae.compile(optimizer='adam', loss=vae_loss)
    return vae, encoder, decoder

vae, encoder, decoder = VAE_models(22)
vae.fit(x_anomaly_train, x_anomaly_train,
        shuffle=True,
        epochs=20,
        batch_size=batch_size,
        validation_data=(x_anomaly_valid, x_anomaly_valid))

if not os.path.exists('vae_data/'):
	os.mkdir('vae_data')

sample_points = np.random.normal(0, 1, (enlarge_data_size, 2))
vae_generate_data = decoder.predict(sample_points)
vae_generate_data_pd = pd.DataFrame(vae_generate_data)
vae_generate_data_pd.to_csv('vae_data/' + str(num_2_kpi[kpi_id]) + '.csv', index=False)



