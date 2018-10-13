import keras as K
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout

def splitInputOutput(data,sequence_length, feature_column_start=0, feature_column_end=-1):
    trainInput = []
    trainOutput = []
    for index in range(len(data) - sequence_length):
        trainInput.append(data[index: index + sequence_length,feature_column_start:feature_column_end])
        trainOutput.append(data[index+sequence_length-1,-1])
    return trainInput, trainOutput

def buildModel(input_shape,drop=0.5,opt='adam',loss='mse',metrics=[]):
    model = Sequential()
    model.add(Bidirectional(LSTM(64,return_sequences=True), input_shape=input_shape))
    model.add(Dropout(drop))
    model.add(Bidirectional(LSTM(128,return_sequences=True)))
    model.add(Dropout(drop))
    model.add(Bidirectional(LSTM(64,return_sequences=False)))
    model.add(Dense(units=256,activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(loss=loss,optimizer=opt,metrics=metrics)
    model.summary()
    return model