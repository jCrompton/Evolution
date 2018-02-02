import math

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, LSTM, Bidirectional, GlobalMaxPool1D, Embedding
from keras.callbacks import TerminateOnNaN, EarlyStopping
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import normalize, scale, minmax_scale

from bitstring import BitArray

def simple_recurrent(predictors, maxlen=80, neural_struc=None, max_features=50000, embedding_dims=150, dropout=None, neural_activation='relu', optimizer='adam'):
    advanced_activation = True if type(neural_activation) != str else False
    # Set hyperparameters if given else use defaults
    units = neural_struc[0][0]/2 if neural_struc else 25
    neural_struct = neural_struc if neural_struc else [[50, 0.1]]

    # Create model with hyperparameters
    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(Bidirectional(LSTM(units, return_sequences=True)))
    model.add(GlobalMaxPool1D())
    if dropout:
        model.add(Dropout(dropout))
    for layer in neural_struct:
        try:
            num_neurons, layer_dropout = layer
        except ValueError:
            num_neurons = layer[0]
            layer_dropout = dropout
        if advanced_activation:
            model.add(Dense(num_neurons))
            model.add(neural_activation)
        else:
            model.add(Dense(num_neurons, activation=neural_activation))
        model.add(Dropout(layer_dropout))
    model.add(Dense(predictors, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def transcribe_base_rnn(sequence, max_num_layers, max_num_neurons, activations=None, optimizers=None, dropouts=None, scalers=None, min_max_features=10000, max_max_features=100000, min_embedding_dims=40, max_embedding_dims=250, min_batch=16, max_batch=128, min_maxlen=40 ,max_maxlen=240, gene_len=False):
    activations = activations if activations else ['relu', 'tanh', 'sigmoid',
                                                   'elu', 'linear', LeakyReLU(),
                                                   'selu', 'hard_sigmoid']
    optimizers = optimizers if optimizers else ['RMSprop', 'adam', 'nadam', 'SGD']
    dropouts = dropouts if dropouts else [0.0, 0.01, 0.1, 0.25, 0.333, 0.5, 0.6, 0.75]
    scalers = scalers if scalers else [None, normalize, minmax_scale, scale]
    len_of_layer = int(math.log(max_num_layers, 2))
    len_of_hidden_arch = sum([int(math.log(max_num_neurons, 2)) - i for i in range(max_num_layers)])
    len_of_dropout = int(math.log(len(dropouts), 2))
    len_of_optimizer = int(math.log(len(optimizers), 2))
    len_of_activation = int(math.log(len(activations), 2))
    len_of_batch_size = int(math.log(max_batch, 2) - math.log(min_batch, 2))
    len_of_scalers = int(math.log(len(scalers), 2))
    len_of_embedding = int(math.log(max_embedding_dims, 2) - math.log(min_embedding_dims, 2))
    len_of_max_features = int(math.log(max_max_features, 2) - math.log(min_max_features, 2))
    len_of_maxlen = int(math.log(max_maxlen, 2) - math.log(min_maxlen, 2))
    genome = [len_of_layer, len_of_hidden_arch, len_of_dropout, len_of_optimizer,
              len_of_activation, len_of_batch_size,
              len_of_scalers, len_of_embedding, len_of_max_features,
              len_of_maxlen]
    if gene_len:
        return sum(genome)
    assert len(sequence) == sum(genome), "Length of input sequence must match length of desired individual:{}, given length {}.".format(sum(genome), len(sequence))

    # Generate hidden architecture
    num_layers = transcribe(sequence, genome, 0)+1
    hidden_architecture = []
    idx = len_of_layer
    window = int(math.log(max_num_neurons, 2))
    for _ in range(num_layers):
        num_of_neurons = BitArray(sequence[idx:idx+window]).uint
        hidden_architecture.append((num_of_neurons, ))
        idx += window
        window -= 1
    dropout = dropouts[transcribe(sequence, genome, 2)]
    optimizer = optimizers[transcribe(sequence, genome, 3)]
    activation = activations[transcribe(sequence, genome, 4)]
    batch_size = (transcribe(sequence, genome, 5)+1)*min_batch
    scaler = scalers[transcribe(sequence, genome, 6)]
    embedding_size = (transcribe(sequence, genome, 7)+1)*min_embedding_dims
    max_features = (transcribe(sequence, genome, 8)+1)*min_max_features
    maxlen = (transcribe(sequence, genome, 9)+1)*min_maxlen
    return genome, hidden_architecture, dropout, optimizer ,activation, batch_size, scaler, embedding_size, max_features, maxlen

def transcribe(sequence, genome, idx):
    start = sum(genome[:idx])
    end = sum(genome[:idx]) + genome[idx]
    return BitArray(sequence[start:end]).uint

def process_data(X_train, X_test, max_features=50000, maxlen=80):
    X_train = [string[0] for string in X_train]
    X_test = [string[0] for string in X_test]
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_test) + list(X_train))
    x_train = tokenizer.texts_to_sequences(X_train)
    x_test = tokenizer.texts_to_sequences(X_test)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return x_train, x_test
