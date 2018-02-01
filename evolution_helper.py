import math

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, PReLU, LSTM, Bidirectional, Convolution1D, Flatten, GlobalMaxPool1D, MaxPooling1D
from keras import backend as K
from keras.callbacks import TerminateOnNaN, EarlyStopping

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize, scale, minmax_scale
from bitstring import BitArray

random_seed = 42
np.random.seed(random_seed)

def build_base_nn_eval_func(x_train, y_train, x_test, y_test, max_num_layers=None,
                            max_num_neurons=None, verbosity=0, epochs=1, *args, **kwargs):
    max_num_layers = max_num_layers if max_num_neurons else 4
    max_num_neurons = max_num_neurons if max_num_neurons else 1024
    dropouts = kwargs.get('dropouts')
    activations = kwargs.get('activations')
    optimizers = kwargs.get('optimizers')
    metrics = kwargs.get('metrics')
    scalers = kwargs.get('scalers')
    verbosity = kwargs.get('verbosity')
    metric = kwargs.get('metric')
    k_folds = kwargs.get('k_folds') if kwargs.get('k_folds') else 3
    shape = x_train.shape[1]
    predictors = y_train.shape[1]
    epochs = epochs

    def base_eval_func(sequence, ret=False, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, k_folds=k_folds, verbose=verbosity, metric=metric, epochs=epochs, gene_len=False):
        if gene_len:
            return transcribe_base(sequence, max_num_layers, max_num_neurons, activations=activations, dropouts=dropouts, optimizers=optimizers, scalers=scalers, gene_len=True)
        genome, hidden_architecture, dropout, optimizer, activation, batch_norm, batch_size, scaler = transcribe_base(sequence, max_num_layers, max_num_neurons, activations=activations, dropouts=dropouts, optimizers=optimizers, scalers=scalers)
        metric = metric if metric else mean_squared_error
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
        scores = []
        X = np.concatenate((x_train,x_test))
        Y = np.concatenate((y_train, y_test))

        for train, test in kfold.split(X, Y):
            if scaler:
                x_train = scaler(x_train)
                x_test = scaler(x_test)
            model = simple_nn(x_train.shape[1], y_train.shape[1], hidden_architecture, dropout, optimizer, activation, batch_norm, batch_size, scaler)
            model.fit(x_train, y_train, epochs=epochs, verbose=verbose, callbacks=[TerminateOnNaN()])
            if ret:
                return model
            pred = model.predict(x_test)

            try:
                scores.append(metric(y_test, pred))
            except ValueError:
                scores.append(999999999)
        return np.mean(scores),
    return base_eval_func

def transcribe_base(sequence, max_num_layers, max_num_neurons, min_batch=16, max_batch=128, activations=None, dropouts=None, optimizers=None, scalers=None, gene_len=False, *args, **kwargs):
    activations = activations if activations else ['relu', 'tanh', 'sigmoid', 'elu', 'linear', LeakyReLU(), 'selu', 'hard_sigmoid']
    optimizers = optimizers if optimizers else ['RMSprop', 'adam', 'nadam', 'SGD']
    dropouts = dropouts if dropouts else [0.0, 0.01, 0.1, 0.25, 0.333, 0.5, 0.6, 0.75]
    scalers = scalers if scalers else [None, normalize, minmax_scale, scale]
    len_of_layer_gene = int(math.log(max_num_layers, 2))
    len_of_hidden_arch_gene = sum([int(math.log(max_num_neurons,
     2)) - i for i in range(max_num_layers)])
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
    idx = max_num_layers
    window = int(math.log(max_num_neurons, 2))
    for _ in range(num_layers):
        num_of_neurons = BitArray(sequence[idx:idx+window]).uint
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

def transcribe(sequence, end, start=0):
    return BitArray(sequence[start:end]).uint

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

def convolutional(maxlen=80, convolutional_structure=None, neural_network_structure=None, max_features=50000, conv_to_dnn_dropout=None, convolutional_activation='relu', neural_activation='relu', embedding_dims=120, max_pooling=False, pool_size=2):
    # Set hyperparameters if given else use defaults
    convolutional_structure = convolutional_structure if convolutional_structure else [(64,3), (32, 3), (16,3)]
    neural_network_structure = neural_network_structure if neural_network_structure else [(180,0.2), (90, 0.2)]

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
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def simple_recurrent(maxlen=80, neural_struc=None, max_features=50000, embedding_dims=150, dropout=None, neural_activation='relu'):
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
        model.add(Dense(num_neurons, activation=neural_activation))
        model.add(Dropout(layer_dropout))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
