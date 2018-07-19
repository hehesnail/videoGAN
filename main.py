import os
import scipy.misc
import tensorflow as tf
import numpy as numpy
import pprint
from model import VGAN
from utils import show_all_variables, visualize

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Training epochs")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term for adam")
flags.DEFINE_integer("zdim", 100, "Dimension of the latent variable")
flags.DEFINE_float("lam", 0.0, "sparsity regularizer")
flags.DEFINE_integer("batch_size", 64, "Batch size for training")
flags.DEFINE_integer("input_height", 64, "The height of input image to use")
flags.DEFINE_integer("input_width", None, "The width of the input image, None will be same as height")
flags.DEFINE_integer("output_height", 64, "The height of the output image")
flags.DEFINE_integer("output_width", None, "The width of output image, None: the same as height")
flags.DEFINE_integer("depth", 32, "Frames to be generated")
flags.DEFINE_string("dataset", "ovp", "The name of the dataset")
flags.DEFINE_string("input_fname_pattern", "*.jpeg", "Glob patterm of filename of images")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Dir for the model checkpoint")
flags.DEFINE_string("sample_dir", "samples", "Dir to save generated images and videos")
flags.DEFINE_boolean("train", False, "True to train the model")
flags.DEFINE_boolean("crop", False, "True to center crop the input images")
FLAGS = flags.FLAGS

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width == None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width == None:
        FLAGS.output_width = FLAGS.output_height
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        vgan = VGAN(sess, depth=FLAGS.depth, input_height=FLAGS.input_height,
                    input_width=FLAGS.input_width, output_height=FLAGS.output_height,
                    output_width=FLAGS.output_width, batch_size=FLAGS.batch_size,
                    dataset=FLAGS.dataset, z_dim=FLAGS.zdim, lam=FLAGS.lam,
                    input_fname_pattern=FLAGS.input_fname_pattern,
                    crop=FLAGS.crop, checkpoint_dir=FLAGS.checkpoint_dir,
                    sample_dir=FLAGS.sample_dir)
        
        show_all_variables()

        if FLAGS.train:
            vgan.train(FLAGS)
        else:
            if not vgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train the model first")
        
        OPTION = 1
        visualize(sess, vgan, FLAGS, OPTION)
