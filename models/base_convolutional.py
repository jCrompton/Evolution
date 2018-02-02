import math

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, BatchNormalization, LeakyReLU, Convolution1D, Flatten, MaxPooling1D
from keras.callbacks import TerminateOnNaN, EarlyStopping

from sklearn.preprocessing import normalize, scale, minmax_scale
from bitstring import BitArray

__all__ = ['transcribe_base_cnn', 'base_convolutional']


def transcribe_base_cnn(sequence, max_num_conv_layers=4, max_num_conv_neurons=256, max_kernel_size=6, max_pool_size=4, max_num_neural_layers=4, max_num_neural_neurons=1024, activations=None, optimizers=None, dropouts=None, scalers=None, min_max_features=10000, max_max_features=100000, min_embedding_dims=40, max_embedding_dims=250, min_batch=16, max_batch=128, min_maxlen=40 ,max_maxlen=240, gene_len=False):
    activations = activations if activations else ['relu', 'tanh', 'sigmoid',
                                                   'elu', 'linear', LeakyReLU(),
                                                   'selu', 'hard_sigmoid']
    optimizers = optimizers if optimizers else ['RMSprop', 'adam', 'nadam', 'SGD']
    dropouts = dropouts if dropouts else [0.0, 0.01, 0.1, 0.25, 0.333, 0.5, 0.6, 0.75]
    scalers = scalers if scalers else [None, normalize, minmax_scale, scale]
    len_of_conv_layer = int(math.log(max_num_conv_layers, 2))
    len_of_conv_arch = sum([int(math.log(max_num_conv_neurons, 2)) - i for i in range(max_num_conv_layers)])
    len_max_kernel = int(math.log(max_kernel_size, 2))
    len_max_pool_size = int(math.log(max_pool_size, 2))
    len_of_neural_layer = int(math.log(max_num_neural_layers, 2))
    len_of_neural_arch = sum([int(math.log(max_num_neural_neurons, 2)) - i for i in range(max_num_neural_layers)])
    len_of_conv_dropout = int(math.log(len(dropouts), 2))
    len_of_neural_dropout = int(math.log(len(dropouts), 2))
    len_of_optimizer = int(math.log(len(optimizers), 2))
    len_of_conv_activation = int(math.log(len(activations), 2))
    len_of_neural_activation = int(math.log(len(activations), 2))
    len_of_batch_size = int(math.log(max_batch, 2) - math.log(min_batch, 2))
    len_of_scalers = int(math.log(len(scalers), 2))
    len_of_embedding = int(math.log(max_embedding_dims, 2) - math.log(min_embedding_dims, 2))
    len_of_max_features = int(math.log(max_max_features, 2) - math.log(min_max_features, 2))
    len_of_maxlen = int(math.log(max_maxlen, 2) - math.log(min_maxlen, 2))
    len_of_max_pool = 1
    genome = [len_of_conv_layer, len_of_conv_arch, len_max_kernel,
              len_max_pool_size, len_of_neural_layer,
              len_of_neural_arch, len_of_conv_dropout, len_of_neural_dropout,
              len_of_optimizer, len_of_conv_activation, len_of_neural_activation,
              len_of_batch_size, len_of_scalers, len_of_embedding,
              len_of_max_features, len_of_maxlen, len_of_max_pool]
    if gene_len:
        return sum(genome)
    assert len(sequence) == sum(genome), "Length of input sequence must match length of desired individual:{}, given length {}.".format(sum(genome), len(sequence))

    # Generate convolutional hidden architecture
    num_conv_layers = transcribe(sequence, genome, 0)+1
    conv_architecture = []
    idx = len_of_conv_layer
    window = int(math.log(max_num_conv_neurons, 2))
    kernel_size = transcribe(sequence, genome, 2) + 1
    pool_size = transcribe(sequence, genome, 3) + 1
    for _ in range(num_conv_layers):
        num_of_neurons = BitArray(sequence[idx:idx+window]).uint
        conv_architecture.append((num_of_neurons, kernel_size))
        idx += window
        window -= 1

    # Generate neural hidden architecture
    num_neural_layers = transcribe(sequence, genome, 2) + 1
    neural_architecture = []
    idx = sum(genome[:2])
    window = int(math.log(max_num_neural_neurons, 2))
    neural_dropout = dropouts[transcribe(sequence, genome, 7)]
    for _ in range(num_neural_layers):
        num_of_neurons = BitArray(sequence[idx:idx+window]).uint
        neural_architecture.append((num_of_neurons, neural_dropout))
        idx += window
        window -= 1

    conv_dropout = dropouts[transcribe(sequence, genome, 6)]
    neural_dropout = dropouts[transcribe(sequence, genome, 7)]
    optimizer = optimizers[transcribe(sequence, genome, 8)]
    conv_activation = activations[transcribe(sequence, genome, 9)]
    neural_activation = activations[transcribe(sequence, genome, 10)]
    batch_size = (transcribe(sequence, genome, 11)+1)*min_batch
    scaler = scalers[transcribe(sequence, genome, 12)]
    embedding_size = (transcribe(sequence, genome, 13)+1)*min_embedding_dims
    max_features = (transcribe(sequence, genome, 14)+1)*min_max_features
    maxlen = (transcribe(sequence, genome, 15)+1)*min_maxlen
    max_pool = [True, False][transcribe(sequence, genome, 16)]
    return genome, conv_architecture, pool_size, neural_architecture, conv_dropout, neural_dropout, optimizer, conv_activation, neural_activation, batch_size, scaler, embedding_size, max_features, maxlen, max_pool


def base_convolutional(predictors, maxlen=80, convolutional_structure=None,
                       neural_network_structure=None, max_features=50000,
                       conv_to_dnn_dropout=None, convolutional_activation='relu',
                       neural_activation='relu', neural_dropout=0.2,
                       embedding_dims=120, max_pooling=False, pool_size=2, optimizer='adam'):
    # Set hyperparameters if given else use defaults
    convolutional_structure = convolutional_structure if convolutional_structure else [(64, 3), (32, 3), (16, 3)]
    neural_network_structure = neural_network_structure if neural_network_structure else [(180,neural_dropout), (90, neural_dropout)]

    # Create model with hyperparameters
    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    for filters, kernel_size in convolutional_structure:
        model.add(Convolution1D(filters, kernel_size, activation=convolutional_activation))
        if max_pooling:
            model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    if conv_to_dnn_dropout:
        model.add(Dropout(conv_to_dnn_dropout))
    for neurons, nn_drop in neural_network_structure:
        model.add(Dense(neurons, activation=neural_activation))
        model.add(Dropout(nn_drop))
    model.add(Dense(predictors, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def transcribe(sequence, genome, idx):
    start = sum(genome[:idx])
    end = sum(genome[:idx]) + genome[idx]
    return BitArray(sequence[start:end]).uint
