import math
import random
import scipy.misc
import numpy as np
from glob import glob
import os
from time import gmtime, strftime
import skvideo.io
import skimage.transform
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

def read_and_process_video(files,size,nof):
    videos = np.zeros((size,nof,64,64,3))
    counter = 0
    for file in files:
        vid = skvideo.io.vreader(file)
        curr_frames = []
        i = 0
        for frame in vid:
            ## Considering first 10 frames for now.
            frame = skimage.transform.resize(frame,[64,64])
            #if len(frame.shape)<3:
            #    frame = np.repeat(frame,3).reshape([64,64,3])
            curr_frames.append(frame)
            i = i + 1
        curr_frames = np.array(curr_frames)
        curr_frames = curr_frames*255.0
        curr_frames = curr_frames/127.5 - 1
        videos[counter,:,:,:,:] = curr_frames
        counter = counter + 1
    return videos


def process_and_write_video(videos,name, save_path):
    videos =np.array(videos)
    videos = np.reshape(videos,[-1,32,64,64,3])
    vidwrite = np.zeros((32,64,64,3))
    for i in range(videos.shape[0]):
        vid = videos[i,:,:,:,:]
        vid = (vid + 1)*127.5
        for j in range(vid.shape[0]):
            frame = vid[j,:,:,:]
            vidwrite[j,:,:,:] = frame
    skvideo.io.vwrite(save_path+ '/' + name + ".mp4",vidwrite)

def visualize(sess, vgan, config, option):
    pass