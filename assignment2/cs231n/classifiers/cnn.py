from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        pass
        # prepare the variables that i need
        C, H, W = input_dim
        # W1 shape = (F, C, HH, WW)
        W1_shape = (num_filters, C, filter_size, filter_size)
        b1_shape = (num_filters)
        # conv means : H ->  (H after conv)  -> (H after pool) = HH
        # see below in the loss part to find the stride of 1 and the pad of (filter_size - 1)//2
        H_after_conv = H   #H - filter_size + ( (filter_size - 1 ) // 2 ) * 2 
        W_after_conv = W
        # 2 x 2 max-pool layer
        pool_size =  2 
        pool_stride = 2 # find below in the loss part
        HH = (H_after_conv - pool_size) // pool_stride + 1
        WW = (W_after_conv - pool_size) // pool_stride + 1
        D = num_filters * HH * WW
        W2_shape = (D, hidden_dim)
        b2_shape = (hidden_dim)
        W3_shape = (hidden_dim, num_classes)
        b3_shape = (num_classes)
        # do the initialization
        self.params["W1"] = weight_scale * np.random.randn(*W1_shape)
        self.params["b1"] = np.zeros(b1_shape)
        self.params["W2"] = weight_scale * np.random.randn(*W2_shape)
        self.params["b2"] = np.zeros(b2_shape)
        self.params["W3"] = weight_scale * np.random.randn(*W3_shape)
        self.params["b3"] = np.zeros(b3_shape)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        pass
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out2, cache2 = affine_relu_forward(out1, W2, b2)
        scores, cache3 = affine_forward(out2,W3,b3)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        pass
        loss, dupstream = softmax_loss(scores, y)
        loss += 0.5*self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3) ) 

        #grads
        dupstream, grads["W3"], grads["b3"] = affine_backward(dupstream, cache3)
        dupstream, grads["W2"], grads["b2"] = affine_relu_backward(dupstream, cache2)
        dupstream, grads["W1"], grads["b1"] = conv_relu_pool_backward(dupstream,cache1)

        grads["W1"] += self.reg * self.params["W1"]
        grads["W2"] += self.reg * self.params["W2"]
        grads["W3"] += self.reg * self.params["W3"]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
