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
from keras.models import load_model

def load_data(sequences):
    sequences = np.load(sequences)
    return sequences

def main():
    sequences = sys.argv[1]
    model_files = sys.argv[2]
    
    predict_labels_name = model_files.replace(".h5","")+sequences.replace(".matrix.npy","")
    sequences = load_data(sequences)

    model = load_model(model_files)
    predicted_labels = model.predict(sequences, batch_size=32)

    np.save(predict_labels_name, predicted_labels)
    
    
if __name__=="__main__":
    main()
