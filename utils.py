import math
import random
import scipy.misc
import numpy as np
from glob import glob
import os
from time import gmtime, strftime

import tensorflow as tf
import tensorflow.contrib.slim as slim

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def load_data(dataset_name, f_pattern):
    return glob(os.path.join("./data", dataset_name, f_pattern))

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def get_image(image_path, resize_h=64, resize_w=64):
    image = imread(image_path)
    return transform(image, resize_h, resize_w)

def transform(image, resize_h=64, resize_w=64):
    cropped_image = scipy.misc.imresize(image, [resize_h, resize_w])
    return np.array(cropped_image) / 127.5 - 1.


def visualize(sess, vgan, config, option):
    pass