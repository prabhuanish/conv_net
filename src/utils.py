from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy
import pickle
import gzip

import theano
import theano.tensor as T

def load_data(dataset, augment_data = False):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    if (dataset == 'mnist'):
        train_set, valid_set, test_set = load_mnist_data("mnist.pkl.gz")

    elif (dataset == 'cifar-10'):
        train_set, valid_set, test_set = load_cifar_data("cifar-10-batches-py", augment_data)   

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

# Check if we have the mnist data set, if not load it
def load_mnist_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
            
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            return pickle.load(f, encoding='latin1')
        except:
            return pickle.load(f)

# We will assume that the cifar data set is already loaded
# Dataset Source: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
def load_cifar_data(dataset, augment_data):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'cifar-10-batches-py':
            dataset = new_path

    print('... loading data')
    
    # retrieve train set (data batch 1)
    d_tr = unpickle(dataset + '/data_batch_1')
    x_train = d_tr['data']
    y_train = d_tr['labels']

    # retrieve rest of train set (batch 2 - 4)
    for i in range(3):
        d_tr = unpickle(dataset + '/data_batch_' + str(i+2))
        x_train = numpy.concatenate((x_train, d_tr['data']), axis=0)
        y_train = numpy.concatenate((y_train, d_tr['labels']), axis=0)

    # Augment training data by mirroring it
    if (augment_data):
        print("Adding mirror images")
        x_mirror_train = x_train[:,::-1]
        x_train = numpy.concatenate((x_train, x_mirror_train), axis=0)
        y_train = numpy.concatenate((y_train,y_train),axis=0)

        print("Shuffle images")
        x_train, y_train = shuffle_data(x_train, y_train)

    train = (x_train, y_train)



    # retrieve valid set (data batch 2)
    d_v = unpickle(dataset + '/data_batch_5')
    x_valid = d_v['data']
    y_valid = d_v['labels']
    valid = (x_valid, y_valid)

    # retrieve valid set (data batch 2)
    d_t = unpickle(dataset + '/test_batch')
    x_test = d_t['data']
    y_test = d_t['labels']
    test = (x_test, y_test)

    return train, valid, test

# Shuffle data and labels in unison
def shuffle_data(data, labels):
    assert len(data) == len(labels)
    shuffled_data = numpy.empty(data.shape, dtype=data.dtype)
    shuffled_labels = numpy.empty(labels.shape, dtype=labels.dtype)

    rand = numpy.random.permutation(len(data))
    for old_ind, new_ind in enumerate(rand):
        shuffled_data[new_ind] = data[old_ind]
        shuffled_labels[new_ind] = labels[old_ind]
    return shuffled_data, shuffled_labels

# Unpickle a file in python (specifically for cifar dataset)
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict
