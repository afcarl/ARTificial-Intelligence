#!/usr/bin/env python2

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import sys


def main():
    which_data = sys.argv[1]

    layer_weights = np.load("weights_{0}.numpy".format(which_data))
    layer_weights = layer_weights.T

    (rows, cols) = (5, 4)
    plt.figure(figsize = (16, 24))
    gs = gridspec.GridSpec(rows, cols, wspace = 0.1, hspace = 0.1)

    for i in range(layer_weights.shape[0]):
        row = i // cols
        col = i % cols
        ax = plt.subplot(gs[row, col])
        nothing = plt.axis("off")
        nothing = plt.imshow(layer_weights[i].T, interpolation = "nearest")

    plt.savefig("filters_{0}.png".format(which_data))
