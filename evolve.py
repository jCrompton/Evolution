import os
import argparse
import time
import names

import numpy as np
import pandas as pd

from scipy.stats import bernoulli
from bitstring import BitArray

from deap import base, creator, tools, algorithms

from evolution_helper import build_base_nn_eval_func

np.random.seed(42)

def get_data(data_path):
    x_train = pd.read_csv('{}/x_train.csv'.format(data_path)).values
    y_train = pd.read_csv('{}/x_train.csv'.format(data_path)).values
    x_test = pd.read_csv('{}/x_train.csv'.format(data_path)).values
    y_test = pd.read_csv('{}/x_train.csv'.format(data_path)).values
    return x_train, y_train, x_test, y_test

def fail_safe(start_time, end_time, tools, population, k_to_save, eval_func, frozen_models_path):
    text = raw_input('\n Do you want to save best so far? (y/n)\n')
    if 'y' in text:
        print('Scoring best so far...')
        best_individuals = tools.selBest(population, k=k_to_save)
        best_evolutionary_score = eval_func(best_individuals[0])
        print('Best score {} in {} minutes{}\n'.format(best_evolutionary_score, end_time, '!' if best_evolutionary_score < 1.0 else '...'))
        print('Now saving top {} models to {}...'.format(k_to_save, frozen_models_path))
        for dna, name in zip(best_individuals, [names.get_full_name() for _ in range(k_to_save)]):
            model = eval_func(dna, ret=True, epochs=build_epochs)
            save_model(model, frozen_models_path, name)
    elif 'n' in text:
        print('\nBye')
    else:
        fail_safe(start_time, end_time, tools, population, k_to_save, eval_func, frozen_models_path)

def save_model(model, path, model_name):
    model_path = '{}/{}'.format(path, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_json = model.to_json()
    with open('{}/frozen_params.json'.format(model_path), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('{}/frozen_weights.h5'.format(model_path))

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_layers', action='store', default=4)
    parser.add_argument('--max_neurons', action='store', default=1024)
    parser.add_argument('--population_size', action='store', default=15)
    parser.add_argument('--data_path', action='store')
    parser.add_argument('--generations', action='store', default=25)
    parser.add_argument('--gene_mutation', action='store', default=0.2)
    parser.add_argument('--gene_crossover', action='store', default=0.4)
    parser.add_argument('--verbosity', action='store', default=1)
    parser.add_argument('--epochs', action='store', default=1)
    parser.add_argument('--save_k_best', action='store', default=1)
    parser.add_argument('--build_epochs', action='store', default=10)
    parser.add_argument('--frozen_models_path', action='store', default=None)
    return parser

def build_base_toolbox(pop_size, eval_func):
    creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
    creator.create('Individual', list , fitness = creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('binary', bernoulli.rvs, 0.5)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary,
    n = gene_length)
    toolbox.register('population', tools.initRepeat, list , toolbox.individual)

    toolbox.register('mate', tools.cxOrdered)
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('evaluate', eval_func)

    population = toolbox.population(n = pop_size)

    return creator, toolbox, population

if __name__ == '__main__':
    start_time = time.time()
    parser = build_parser()
    arguments = parser.parse_args()
    # Gather variables
    data_path = str(arguments.data_path)
    max_layers = int(arguments.max_layers)
    max_neurons = int(arguments.max_neurons)
    pop_size = int(arguments.population_size)
    generations = int(arguments.generations)
    gene_mut = float(arguments.gene_mutation)
    gene_crsv = float(arguments.gene_crossover)
    verbosity = int(arguments.verbosity)
    epochs = int(arguments.epochs)
    k_to_save = int(arguments.save_k_best)
    build_epochs = int(arguments.build_epochs)
    frozen_models_path = arguments.frozen_models_path if arguments.frozen_models_path else data_path

    # Check arguments are clean
    assert os.path.exists(data_path), 'Make sure --data_path argument points to an existing directory. Given directory {}.'.format(data_path)
    assert gene_mut <= 1.0, 'Gene mutation cannot be greater than 1...'
    assert gene_crsv <= 1.0, 'Gene crossover cannot be greater than 1...'
    assert os.path.exists(frozen_models_path), 'Make sure frozen_models_path points to an existing directory. Given path {}'.format(frozen_models_path)
    # Get data
    x_train, y_train, x_test, y_test = get_data(data_path)
    # Build toolbox and model function
    eval_func = build_base_nn_eval_func(x_train, y_train, x_test, y_test, max_num_layers=max_layers, max_num_neurons=max_neurons, verbosity=verbosity, epochs=epochs)
    gene_length = eval_func([1], gene_len=True)

    creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
    creator.create('Individual', list , fitness = creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('binary', bernoulli.rvs, 0.5)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary,
    n = gene_length)
    toolbox.register('population', tools.initRepeat, list , toolbox.individual)

    toolbox.register('mate', tools.cxOrdered)
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('evaluate', eval_func)

    population = toolbox.population(n=pop_size)

    try:
        algorithms.eaSimple(population, toolbox,cxpb=gene_crsv, mutpb=gene_mut,
        ngen=generations, verbose=False if verbosity < 1 else True)

        end_time = round((time.time() - start_time)/60.0, 3)
        best_individuals = tools.selBest(population, k=k_to_save)
        best_evolutionary_score = eval_func(best_individuals[0])
        print('Evolution complete, best score {} in {} minutes{}\n'.format(best_evolutionary_score, end_time, '!' if best_evolutionary_score < 1.0 else '...'))
        print('Now saving top {} models to {}...'.format(k_to_save, frozen_models_path))
        for dna, name in zip(best_individuals, [names.get_full_name() for _ in range(k_to_save)]):
            model = eval_func(dna, ret=True, epochs=build_epochs)
            save_model(model, frozen_models_path, name)
    except KeyboardInterrupt:
        end_time = round((time.time() - start_time)/60.0, 3)
        fail_safe(start_time, end_time, tools, population, k_to_save, eval_func, frozen_models_path)
