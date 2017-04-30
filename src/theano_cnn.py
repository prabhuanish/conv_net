from __future__ import print_function

import os
import sys
import timeit

import numpy
import pickle

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d


from utils import load_data
from logistic_sgd import LogisticRegression
from hidden_layer import HiddenLayer
from conv_pool_layer import LeNetConvPoolLayer

def evaluate_lenet5(learning_rate=0.1, momentum=0.9, n_epochs=500,
                    dataset='mnist', depth = 1, augment_data = False,
                    nkerns=[20, 50, 100], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type n_feature_maps: int
    :param n_feature_maps: number of feature maps in input, i.e. 3 for RGB image

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset, augment_data)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Set the initial dimensions of the input images
    if (dataset == 'mnist'):
        in_dim = 28;
    elif (dataset == 'cifar-10'):
        in_dim = 32;


    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, depth, in_dim, in_dim))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    filter_dim = 5
    pool_dim = 1

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, depth, in_dim, in_dim),
        filter_shape=(nkerns[0], depth, filter_dim, filter_dim),
        poolsize=(pool_dim, pool_dim)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    in_dim = (in_dim - filter_dim + 1) / pool_dim
    filter_dim = 5
    pool_dim = 1

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], in_dim, in_dim),
        filter_shape=(nkerns[1], nkerns[0], filter_dim, filter_dim),
        poolsize=(pool_dim, pool_dim)
    )

    # Construct the third convolutional layer with no pooling
    # filtering reduces the image size to (4-3+1, 4-3+1) = (2, 2)
    # 4D output tensor is thus of shape (batch_size, nkerns[2], 2, 2)
    in_dim = (in_dim - filter_dim + 1) / pool_dim
    filter_dim = 3
    pool_dim = 1

    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], in_dim, in_dim),
        filter_shape=(nkerns[2], nkerns[1], filter_dim, filter_dim),
        poolsize=(pool_dim, pool_dim)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[2] * 2 * 2),
    # or (500, 200 * 2 * 2) = (500, 800) with the default values.
    layer3_input = layer2.output.flatten(2)
    # construct a fully-connected rectifier layer
    in_dim = (in_dim - filter_dim + 1) / pool_dim

    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * in_dim * in_dim,
        n_out=500,
        activation=relu
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
            
    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=gradient_updates_momentum(cost, params, learning_rate, momentum),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look at this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        learning_rate = set_learning_rate(learning_rate, epoch, dataset)
        momentum = set_momentum(momentum, epoch)

        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)

            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on train set
                train_losses = [train_model(i) for i
                                     in range(n_train_batches)]
                this_train_loss = numpy.mean(train_losses)
                print('     epoch %i, minibatch %i/%i, train error %f %%\n' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_train_loss * 100.))

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('     epoch %i, minibatch %i/%i, validation error %f %%\n' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%\n') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break



    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

# Define rectifier linear unit activation function
def relu(x):
    return theano.tensor.switch(x<0, 0, x)

# Compute updates for gradient descent with momentum 
def gradient_updates_momentum(cost, params, learning_rate, momentum):
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    
    # List of update steps for each parameter
    updates = []
    
    for param in params:
        
        # create a shared variable to track previous param
        previous_step = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        
        # symbolic Theano variable that represents the L1 regularization term
        # L1  = T.sum(abs(param))
        # symbolic Theano variable that represents the squared L2 term
        # L2 = T.sum(param ** 2)
        # Set regularization coeefficients
        # lambda_1, lambda_2 = 0.00001, 0.0000001
        # Add regularization to cost function (not needed for CNN)
        # cost += lambda_1 * L1 + lambda_2 * L2

        step = momentum*previous_step - learning_rate*T.grad(cost, param)
        
        # Add an update to store the previous step value
        updates.append((previous_step, step))
        
        # Add an update to apply the gradient descent step to the parameter itself
        updates.append((param, param + step))

    return updates

# Dynamically set the learning rate to avoid overfitting
def set_learning_rate(learning_rate, epoch, dataset='mnist'):

    # Dynamically update learning rate every 10 epochs
    if epoch % 10 == 0:
        return 0.95 * learning_rate

    # Fine tune learning rate for mnist dataset
    if (dataset == 'mnist'):
        # Lower the learning rate dynamically to avoid overfitting
        if epoch == 2:
            return 0.01
        elif epoch == 3:
            return 0.0001
        elif epoch == 5:
            return 0.000001  
        elif epoch == 7:
            return 0.00000001  


    # Fine tune learning rate for cifar-10 dataset
    elif (dataset == 'cifar-10'):
        '''return 0.01
        if epoch == 4:
            return 0.01
        elif epoch == 32:
            return 0.0001   '''     

    # No change to learning rate
    return learning_rate

# Dynamically set the momentum
def set_momentum(momentum, epoch):
    if epoch % 10 == 0:
        return 1.05 * momentum

    return momentum

if __name__ == '__main__':
    # Check the dataset we are using and launch our CNN
    if (len(sys.argv) == 2 and sys.argv[1] == 'cifar-10'):
        print("Launching CNN with cifar-10 dataset (32x32 RGB images).")
        evaluate_lenet5(depth=3, dataset='cifar-10', augment_data=True, learning_rate=0.001)
    elif (len(sys.argv) == 2 and sys.argv[1] == 'mnist'):
        print("Launching CNN with MNIST dataset (handwritten digits)")
        evaluate_lenet5(learning_rate=0.1)
    else:
        print("Launching CNN with default dataset, MNIST (handwritten digits)")
        evaluate_lenet5(learning_rate=0.1)

