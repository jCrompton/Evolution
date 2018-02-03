import math

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras import backend as K
from keras.callbacks import TerminateOnNaN, EarlyStopping

from sklearn.preprocessing import normalize, scale, minmax_scale
from bitstring import BitArray


def transcribe_base(sequence, max_num_layers, max_num_neurons, min_batch=16, max_batch=128, activations=None, dropouts=None, optimizers=None, scalers=None, gene_len=False):
    activations = activations if activations else ['relu', 'tanh', 'sigmoid', 'elu', 'linear', LeakyReLU(), 'selu', 'hard_sigmoid']
    optimizers = optimizers if optimizers else ['RMSprop', 'adam', 'nadam', 'SGD']
    dropouts = dropouts if dropouts else [0.0, 0.01, 0.1, 0.25, 0.333, 0.5, 0.6, 0.75]
    scalers = scalers if scalers else [None, normalize, minmax_scale, scale]
    len_of_layer_gene = int(math.log(max_num_layers, 2))
    len_of_hidden_arch_gene = sum([int(math.log(max_num_neurons, 2)) - i for i in range(max_num_layers)])
    len_of_dropout_gene = int(math.log(len(dropouts), 2))
    len_of_optimizer_gene = int(math.log(len(optimizers), 2))
    len_of_activation_gene = int(math.log(len(activations), 2))
    len_of_batch_norm_gene = 1
    len_of_batch_size = int(math.log(max_batch, 2) - math.log(min_batch, 2))
    len_of_scalers = int(math.log(len(scalers), 2))
    genome = [len_of_layer_gene, len_of_hidden_arch_gene,
                      len_of_dropout_gene, len_of_optimizer_gene,
                      len_of_activation_gene, len_of_batch_norm_gene,
                      len_of_batch_size, len_of_scalers]
    if gene_len:
        return sum(genome)
    assert len(sequence) == sum(genome), "Length of input sequence must match length of desired individual:{}, given length {}.".format(sum(genome), len(sequence))

    num_layers = transcribe(sequence, len_of_layer_gene)+1
    # Generate hidden architecture
    hidden_architecture = []
    idx = len_of_layer_gene
    window = int(math.log(max_num_neurons, 2))
    for _ in range(num_layers):
        num_of_neurons = BitArray(sequence[idx:idx+window]).uint + 1
        hidden_architecture.append((num_of_neurons, ))
        idx += window
        window -= 1
    dropout = dropouts[transcribe(sequence, sum(genome[:2])+genome[2], start=sum(genome[:2]))]
    optimizer = optimizers[transcribe(sequence, sum(genome[:3])+genome[3], start=sum(genome[:3]))]
    activation = activations[transcribe(sequence, sum(genome[:4])+genome[4], start=sum(genome[:4]))]
    batch_norm = [True, False][transcribe(sequence, sum(genome[:5])+genome[5], start=sum(genome[:5]))]
    batch_size = (transcribe(sequence, sum(genome[:6])+genome[6], start=sum(genome[:6]))+1)*min_batch
    scaler = scalers[transcribe(sequence, sum(genome[:7])+genome[7], start=sum(genome[:7]))]
    return genome, hidden_architecture, dropout, optimizer, activation, batch_norm, batch_size, scaler


def simple_nn(shape, predictors, hidden_architecture, dropout, optimizer, activation, batch_norm, batch_size, scaler):
    advanced_activation = True if type(activation) != str else False
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(shape,)))
    for layer_params in hidden_architecture:
        layer_size = layer_params[0]
        try:
            hidden_dropout = layer_params[1]
        except IndexError:
            hidden_dropout = dropout
        if advanced_activation:
            model.add(Dense(layer_size))
            model.add(activation)
        else:
            model.add(Dense(layer_size, activation=activation))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(hidden_dropout))
    model.add(Dense(predictors))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def transcribe(sequence, end, start=0):
    return BitArray(sequence[start:end]).uint
