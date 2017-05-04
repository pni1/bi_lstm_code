from __future__ import print_function
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.models import Model

from keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dropout,
    Dense
)

from keras.optimizers import RMSprop
from keras.models import Sequential
import numpy as np
from keras.utils import plot_model

from keras import optimizers
import sys

def load_data(sequences, labels):
    sequence = np.load(sequences)
    label = np.load(labels)

    num_train = (sequence.shape[0]//2)*1

    train_sequence = sequence[0:num_train, :, :]
    train_label = label[0:num_train,]
    train_set = (train_sequence, train_label)

    valid_sequence = sequence[num_train+1:,:,:]
    valid_label = label[num_train+1:,]
    valid_set = (valid_sequence, valid_label)
    
    return train_set, valid_set

def main():
    sequences = sys.argv[1]
    labels = sys.argv[2]
    n_epochs = int(sys.argv[3])

    logger_file = sequences.replace("_matrix.npy", "_"+str(n_epochs)+".log")
    model_name = sequences.replace("_matrix.npy", "_"+str(n_epochs)+"_model.h5")

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=1, min_lr=0.5e-6)
    early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=1)
    drn_logger = CSVLogger(logger_file)
    
    train_set, valid_set = load_data(sequences, labels)
    X_train, y_train = train_set
    X_valid, y_valid = valid_set

    timestep = 50
    data_dim = 300
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(timestep, data_dim)))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(2, activation='sigmoid', name='sigmoid'))

    rmsprop = RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])


    print('Train...')
    model.fit(X_train, y_train,
              batch_size=32,
              epochs=n_epochs,
              validation_data=(X_valid, y_valid),
              callbacks=[lr_reducer, early_stopper, drn_logger])

    score, acc = model.evaluate(X_valid, y_valid,
                                batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    model.save(model_name)

    
if __name__=="__main__":
    main()
