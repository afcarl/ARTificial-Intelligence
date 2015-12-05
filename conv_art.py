#!/usr/bin/env python2

"""
This code is modified from the tutorial found at http://deeplearning.net/tutorial/lenet.html.
"""
import numpy as np
import cPickle
import sys
import theano
import theano.tensor as T
import time

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network."""
    
    def __init__(self, rng, input, filter_shape, image_shape, poolsize = (2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.
        
        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)
        
        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        
        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        
        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low = -W_bound, high = W_bound, size = filter_shape),
                dtype = theano.config.floatX
            ),
            borrow=True
        )
        
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype = theano.config.floatX)
        self.b = theano.shared(value = b_values, borrow = True)
        
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input = input,
            filters = self.W,
            filter_shape = filter_shape,
            image_shape = image_shape
        )
        
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input = conv_out,
            ds = poolsize,
            ignore_border = True
        )
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle("x", 0, "x", "x"))
        
        # store parameters of this layer
        self.params = [self.W, self.b]


"""
:type learning_rate: float
:param learning_rate: learning rate used (factor for the stochastic
                      gradient)

:type n_epochs: int
:param n_epochs: maximal number of epochs to run the optimizer

:type nkerns: list of ints
:param nkerns: number of kernels on each layer
"""
learning_rate = 0.1
n_epochs = 200
nkerns = [20, 50, 50]
batch_size = 500

rng = np.random.RandomState(23455)

which_data = sys.argv[1]

train_set = cPickle.load(open("train_set_{0}.numpy".format(which_data)))
valid_set = cPickle.load(open("valid_set_{0}.numpy".format(which_data)))
test_set = cPickle.load(open("test_set_{0}.numpy".format(which_data)))


def shared_dataset(data_xy, borrow = True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    (data_x, data_y) = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype = theano.config.floatX),
                             borrow = borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype = theano.config.floatX),
                             borrow = borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return (shared_x, T.cast(shared_y, "int32"))


(test_set_x, test_set_y) = shared_dataset(test_set)
(valid_set_x, valid_set_y) = shared_dataset(valid_set)
(train_set_x, train_set_y) = shared_dataset(train_set)

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow = True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow = True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow = True).shape[0] / batch_size

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch

x = T.tensor4("x")   # the data is presented as rasterized images
y = T.ivector("y")  # the labels are presented as 1D vector of
                    # [int] labels

######################
# BUILD ACTUAL MODEL #
######################
print("... building the model")

# Reshape matrix of rasterized images of shape (batch_size, RGB [1|3], height, width)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
(RGB, height, width) = train_set[0][0].shape
layer0_input = x.reshape((batch_size, RGB, height, width))

layer0_filter_height = 5
layer0_filter_width = 5

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (100 - 5 + 1 , 100 - 5 + 1) = (96, 96)
# Why + 1? Draw a picture. It makes sense.
# maxpooling reduces this further to (96 / 2, 96 / 2) = (48, 48)
# 4D output tensor is thus of shape (batch_size, nkerns[0], 48, 48)
layer0 = LeNetConvPoolLayer(
    rng,
    input = layer0_input,
    image_shape = (batch_size, RGB, height, width),
    filter_shape = (nkerns[0], RGB, layer0_filter_height, layer0_filter_width),
    poolsize = (2, 2)
)

layer0_output_height = (height - layer0_filter_height + 1) / 2
layer0_output_width = (width - layer0_filter_width + 1) / 2

layer1_filter_height = 5
layer1_filter_width = 5

# Construct the second convolutional pooling layer
# filtering reduces the image size to (48 - 5 + 1, 48 - 5 + 1) = (44, 44)
# maxpooling reduces this further to (44 / 2, 44 / 2) = (22, 22)
# 4D output tensor is thus of shape (nkerns[0], nkerns[1], 22, 22)
layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape = (batch_size, nkerns[0], layer0_output_height, layer0_output_width),
    filter_shape = (nkerns[1], nkerns[0], layer1_filter_height, layer1_filter_width),
    poolsize = (2, 2)
)

layer1_output_height = (layer0_output_height - layer1_filter_height + 1) / 2
layer1_output_width = (layer0_output_width - layer1_filter_width + 1) / 2

layer2_filter_height = 5
layer2_filter_width = 5

# Construct the second convolutional pooling layer
# filtering reduces the image size to (22 - 5 + 1, 22 - 5 + 1) = (18, 18)
# maxpooling reduces this further to (18 / 2, 18 / 2) = (9, 9)
# 4D output tensor is thus of shape (nkerns[0], nkerns[1], 9, 9)
layer2 = LeNetConvPoolLayer(
    rng,
    input = layer1.output,
    image_shape = (batch_size, nkerns[1], layer1_output_height, layer1_output_width),
    filter_shape = (nkerns[2], nkerns[1], layer2_filter_height, layer2_filter_width),
    poolsize = (2, 2)
)

layer2_output_height = (layer1_output_height - layer2_filter_height + 1) / 2
layer2_output_width = (layer1_output_width - layer2_filter_width + 1) / 2

# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (batch_size, nkerns[2] * 9 * 9),
# or (500, 50 * 9 * 9) = (500, 3600) with the default values.
layer3_input = layer2.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer3 = HiddenLayer(
    rng,
    input = layer3_input,
    n_in = nkerns[2] * layer2_output_height * layer2_output_width,
    n_out = 500,
    activation = T.tanh
)

numClasses = len(set(train_set[1].tolist()))
# classify the values of the fully-connected sigmoidal layer
layer4 = LogisticRegression(input=layer3.output, n_in = 500, n_out = numClasses)

# the cost we minimize during training is the NLL of the model
cost = layer4.negative_log_likelihood(y)

# create a function to compute the mistakes that are made by the model
test_model = theano.function(
    [index],
    layer4.errors(y),
    givens = {
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    [index],
    layer4.errors(y),
    givens = {
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
    updates = updates,
    givens = {
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)
# end-snippet-1

###############
# TRAIN MODEL #
###############
print("... training")
# early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                       # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant
validation_frequency = min(n_train_batches, patience / 2)
                              # go through this many
                              # minibatches before checking the network
                              # on the validation set; in this case we
                              # check every epoch

best_validation_loss = np.inf
best_iter = 0
test_score = 0.0
start_time = time.clock()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        
        iter = (epoch - 1) * n_train_batches + minibatch_index
        
        if iter % 3 == 0:
            end_time = time.clock()
            print("The code has been running for %.2fm" % ((end_time - start_time) / 60.))
        
        if iter % 100 == 0:
            print("training @ iter = {0}".format(iter))
        cost_ij = train_model(minibatch_index)
        
        if (iter + 1) % validation_frequency == 0:
            
            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i in range(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
            print("epoch %i, minibatch %i/%i, validation error %f %%" %
                  (epoch, minibatch_index + 1, n_train_batches,
                   this_validation_loss * 100.0))
            
            end_time = time.clock()
            print("The code has been running for %.2fm" % ((end_time - start_time) / 60.))
            
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
                test_score = np.mean(test_losses)
                print(("     epoch %i, minibatch %i/%i, test error of "
                       "best model %f %%") %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score * 100.0))
        
        if patience <= iter:
            done_looping = True
            break

end_time = time.clock()
print("Optimization complete.")
print("Best validation score of %f %% obtained at iteration %i, "
      "with test performance %f %%" %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
print("The code ran for %.2fm" % ((end_time - start_time) / 60.))

test_labels = test_set[1].tolist()
counts = {}
for label in test_labels:
    counts[label] = counts.get(label, 0) + 1

counts = list(counts.items())
counts.sort(key = lambda artist: artist[1], reverse = True)

baseline = float(counts[0][1]) / float(len(test_labels))

weights = layer0.W.get_value(borrow = True).T
biases = layer0.b.get_value(borrow = True).T
np.save("weights_{0}".format(which_data), weights)
np.save("biases_{0}".format(which_data), biases)