#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import utility
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD
from utility import RMSE

EMBEDDING_DIM = 300
NUM_FILTERS = 50
FILTER_SIZE = 4
NUM_EPOCHS = 26
BATCH_SIZE = 64


def create_conv_model(sequence_length, embeddings):
    print 'CNN setup'

    input_sequences = Input(shape=(sequence_length,), dtype='int32')
    embedding_layer = Embedding(
        len(embeddings),
        EMBEDDING_DIM,
        weights=[embeddings],
        input_length=sequence_length,
        trainable=True
    )(input_sequences)

    convolution_layer = Conv1D(
        NUM_FILTERS,
        FILTER_SIZE,
        activation='relu',
        border_mode='same',
        input_length=sequence_length
    )(embedding_layer)

    max_pooling_layer = MaxPooling1D(sequence_length)(convolution_layer)
    standard_nn_layer = Flatten()(max_pooling_layer)
    predictions = Dense(20)(standard_nn_layer)

    model = Model(input_sequences, predictions)
    sgd = SGD(lr=0.001)
    model.compile(optimizer=sgd, loss=RMSE)

    return model


def load_saved_model(epoch):
    source = 'output/network_model_' + str(epoch - 1) + '.h5'
    return load_model(source, {'RMSE': RMSE})


def train_model(epoch, model, train_set_x, train_set_y, test_set_x, test_set_y):
    print 'Model training at: ' + str(time.strftime('%H:%M:%S', time.gmtime(time.time() + 3600)))

    model.fit(train_set_x, train_set_y, batch_size=BATCH_SIZE, nb_epoch=1, verbose=2)
    model.save('output/network_model_' + str(epoch) + '.h5')

    scores = model.evaluate(test_set_x, test_set_y, batch_size=BATCH_SIZE, verbose=0)
    print 'RMSE on the test set after ' + str(epoch) + ' epochs = ' + str(scores)


def run(load_model_func=None):

    embeddings, reviews_sequences, review_factors, reviews = utility.get_embeddings_and_sequences()

    train_set_x, train_set_y, test_set_x, test_set_y, prediction_set, prediction_items = \
        utility.get_training_test_and_prediction_set(
            reviews_sequences,
            review_factors,
            reviews
        )

    print train_set_x.shape
    print test_set_x.shape
    current_epoch = NUM_EPOCHS
    for i in range(0, 1):
        if load_model_func is not None:
            model = load_model_func(current_epoch)
        else:
            model = create_conv_model(len(reviews_sequences[0]), embeddings)
        train_model(current_epoch, model, train_set_x, train_set_y, test_set_x, test_set_y)

        predictions = model.predict(prediction_set, batch_size=BATCH_SIZE,verbose=0)

        utility.write_factors_predictions(prediction_items, predictions, current_epoch)
        current_epoch += 1


run(load_model_func=load_saved_model)