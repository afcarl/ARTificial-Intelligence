#!/usr/bin/env python

"""
See http://sklearn-theano.github.io/auto_examples/plot_overfeat_layer1_filters.html.

====================================
Visualization of first layer filters
====================================

The first layers of convolutional neural networks often have very "human
interpretable" values, as seen in these example plots. Visually, these filters
are similar to other filters used in computer vision, such as Gabor filters.

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


def make_visual(layer_weights):
    max_scale = layer_weights.max(axis = -1).max(axis = -1)[...,
                                                        np.newaxis, np.newaxis]
    min_scale = layer_weights.min(axis = -1).min(axis = -1)[...,
                                                        np.newaxis, np.newaxis]
    return (255 * (layer_weights - min_scale) /
            (max_scale - min_scale)).astype("uint8")


def main():
    layer_weights = np.load("movement_weights_arch1.numpy")
    layer_weights = layer_weights.transpose(3, 2, 1, 0)
    
    lw_shape = layer_weights.shape
    lw = make_visual(layer_weights)
    
    # height * width must equal lw_shape[0].
    height = 4
    width = 5
    lw = lw.reshape(height, width, *lw_shape[1:])
    lw = lw.transpose(0, 3, 1, 4, 2)
    lw = lw.reshape(height * lw_shape[-1], width * lw_shape[-2], lw_shape[1])
    
    mosaic = lw
    plt.imshow(mosaic, interpolation = "nearest")
    plt.show()
    plt.savefig("filters_movements_C1.png")