import math
import random
import scipy.misc
import numpy as np
from time import gmtime, strftime

import tensorflow as tf
import tensorflow.contrib.slim as slim

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def visualize(sess, vgan, config, option):
    pass