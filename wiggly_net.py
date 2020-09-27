import numpy as np
import tensorflow.keras as keras
import tensorflow as tf


def build_model(shape,Conv2D_N=4,Dense_N=1,LSTM_N=1,LR=1e-4):
    model = keras.models.Sequential()
#Conv2D block############################################################################################
    for _ in range(Conv2D_N):
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(32, (2, 2), padding='same'),input_shape=(shape[1], shape[2], shape[3], shape[4])))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(32, (2, 2), padding='same')))
        model.add(keras.layers.TimeDistributed(keras.layers.Activation('relu')))
        model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')))
########################################################################################################        
    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    for _ in range(Dense_N):
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(512)))
        model.add(keras.layers.TimeDistributed((keras.layers.Activation('relu'))))
#LSTM block#############################################################################################
    for _ in range(LSTM_N):
        model.add(keras.layers.LSTM(units=512, activation='tanh', recurrent_activation='sigmoid', unit_forget_bias=True, dropout=0.0, recurrent_dropout=0, return_sequences=True))
########################################################################################################
    model.add(keras.layers.Dense(shape[2]*shape[3], activation='linear'))
    model.add(keras.layers.TimeDistributed(keras.layers.Reshape((shape[2],shape[3],shape[4]))))
#######################################################################################################
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(loss='mae',optimizer=optimizer,metrics=['mse'])
    return model