import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
from sklearn.model_selection import KFold
from bitstring import BitArray

from keras.callbacks import TerminateOnNaN
from keras.losses import binary_crossentropy

from models.base_nn import transcribe_base, simple_nn
from models.base_rnn import transcribe_base_rnn, simple_recurrent, process_data
from models.base_convolutional import *
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
    recurrent = kwargs.get('rnn') if kwargs.get('rnn') else False
    convolutional = kwargs.get('cnn') if kwargs.get('cnn') else False
    if recurrent:
        def eval_func(sequence, ret=False, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, k_folds=k_folds, verbose=verbosity, metric=metric, epochs=epochs, gene_len=False):
            if gene_len:
                return transcribe_base_rnn(sequence, max_num_layers, max_num_neurons, activations=activations, optimizers=optimizers, dropouts=dropouts, scalers=scalers, gene_len=True)
            genome, hidden_architecture, dropout, optimizer ,activation, batch_size, scaler, embedding_size, max_features, maxlen = transcribe_base_rnn(sequence, max_num_layers, max_num_neurons, activations=activations, optimizers=optimizers, dropouts=dropouts, scalers=scalers, gene_len=False)
            metric = metric if metric else mean_squared_error
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            scores = []
            x_train, x_test = process_data(x_train, x_test, max_features=max_features, maxlen=maxlen)
            X = np.concatenate((x_train,x_test))
            Y = np.concatenate((y_train, y_test))
            for train, test in kfold.split(X, Y):
                y_test = Y[test]
                y_train = Y[train]
                if False:
                    x_train = scaler(X[train])
                    x_test = scaler(X[test])
                else:
                    x_train = X[train]
                    x_test = X[test]
                model = simple_recurrent(y_train.shape[1], maxlen=maxlen,
                                         neural_struc=hidden_architecture,
                                         max_features=max_features,
                                         embedding_dims=embedding_size,
                                         dropout=dropout,
                                         neural_activation=activation,
                                         optimizer=optimizer)
                model.fit(x_train, y_train, epochs=epochs,
                          verbose=1, callbacks=[TerminateOnNaN()], batch_size=batch_size)
                if ret:
                    return model
                pred = model.predict(x_test)

                try:
                    scores.append(metric(y_test, pred))
                except ValueError, IndexError:
                    scores.append(999999999)
            return np.mean(scores),
    elif convolutional:
        def eval_func(sequence, ret=False, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, k_folds=k_folds, verbose=verbosity, metric=metric, epochs=epochs, gene_len=False):
            if gene_len:
                return transcribe_base_cnn(sequence, max_num_layers, max_num_neurons, activations=activations, optimizers=optimizers, dropouts=dropouts, scalers=scalers, gene_len=True)
            genome, conv_architecture, pool_size, neural_architecture, conv_dropout, neural_dropout, optimizer, conv_activation, neural_activation, batch_size, scaler, embedding_size, max_features, maxlen, max_pool = transcribe_base_cnn(sequence, max_num_layers, max_num_neurons, activations=activations, optimizers=optimizers, dropouts=dropouts, scalers=scalers, gene_len=False)
            metric = metric if metric else mean_squared_error
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            scores = []
            x_train, x_test = process_data(x_train, x_test, max_features=max_features, maxlen=maxlen)
            X = np.concatenate((x_train,x_test))
            Y = np.concatenate((y_train, y_test))
            for train, test in kfold.split(X, Y):
                y_test = Y[test]
                y_train = Y[train]
                if False:
                    x_train = scaler(X[train])
                    x_test = scaler(X[test])
                else:
                    x_train = X[train]
                    x_test = X[test]
                model = base_convolutional(y_train.shape[1], maxlen=maxlen,
                                           convolutional_structure=conv_architecture,
                                           neural_network_structure=neural_architecture, max_features=max_features,
                                           conv_to_dnn_dropout=conv_dropout, convolutional_activation=conv_activation,
                                           neural_activation=neural_activation, neural_dropout=neural_dropout,
                                           embedding_dims=embedding_size, max_pooling=max_pool, pool_size=pool_size, optimizer=optimizer)
                model.fit(x_train, y_train, epochs=epochs,
                          verbose=1, callbacks=[TerminateOnNaN()], batch_size=batch_size)
                if ret:
                    return model
                pred = model.predict(x_test)

                try:
                    score = metric(y_test, pred)
                    scores.append(score)
                except ValueError, IndexError:
                    scores.append(999999999)
            print np.mean(scores)
            return np.mean(scores),
    else:
        def eval_func(sequence, ret=False, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, k_folds=k_folds, verbose=verbosity, metric=metric, epochs=epochs, gene_len=False):
            if gene_len:
                return transcribe_base(sequence, max_num_layers,
                                       max_num_neurons, activations=activations,
                                       optimizers=optimizers, dropouts=dropouts,
                                       scalers=scalers, gene_len=True)
            genome, hidden_architecture, dropout, optimizer, activation, batch_norm, batch_size, scaler = transcribe_base(sequence, max_num_layers, max_num_neurons, activations=activations, dropouts=dropouts, optimizers=optimizers, scalers=scalers)
            metric = metric if metric else mean_squared_error
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            scores = []
            X = np.concatenate((x_train, x_test))
            Y = np.concatenate((y_train, y_test))

            for train, test in kfold.split(X, Y):
                y_test = Y[test]
                y_train = Y[train]
                if scaler:
                    x_train = scaler(x_train)
                    x_test = scaler(x_test)
                model = simple_nn(x_train.shape[1], y_train.shape[1],
                                  hidden_architecture, dropout, optimizer,
                                  activation, batch_norm, batch_size, scaler)
                model.fit(x_train[train], y_train[train], epochs=epochs, verbose=verbose,
                          batch_size=batch_size, callbacks=[TerminateOnNaN()])
                if ret:
                    return model
                pred = model.predict(x_test[test])

                try:
                    score = metric(y_test)
                    scores.append(metric(y_test, pred))
                except ValueError:
                    scores.append(999999999)
            return np.mean(scores),
    return eval_func
