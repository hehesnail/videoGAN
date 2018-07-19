import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

class VGAN(object):
    def __init__(self, sess, depth=32, input_height=64,input_width=64, output_height=64,
                output_width=64, batch_size=64, dataset="ovp", z_dim=100, lam = 0.0,
                input_fname_pattern="*.jpeg", crop=True, checkpoint_dir=None,
                sample_dir=None):
        self.sess = sess

        self.depth = depth
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.z_dim = z_dim
        self.lam = lam
        self.video_dim = [self.depth, self.input_height, self.input_width, 3]

        self.batch_size = batch_size
        self.dataset = dataset
        self.input_fname_pattern = input_fname_pattern
        self.crop = crop
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        
    def build_model(self):
        self.frames = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, 3]) #[N,H,W,3]
        self.real_video = tf.placeholder(tf.float32, [None] + self.video_dim) #[N,D,H,W,3]
        self.fake_video, self.foreground, self.background, self.mask, self.gen_reg = self.generator(self.frames)
        self.pro_real, self.real_logits = self.discriminator(self.real_video)
        self.pro_fake, self.fake_logits = self.discriminator(self.fake_video, reuse=True)

        self.d_real_sum = histogram_summary("d_real", self.pro_real)
        self.d_fake_sum = histogram_summary("d_fake", self.pro_fake)

        self.g_loss_pure = -tf.reduce_mean(self.fake_logits)
        self.g_loss = self.g_loss_pure + self.lam*self.gen_reg
        
        self.g_loss_pure_sum = scalar_summary("g_loss_pure", self.g_loss_pure)
        self.gen_reg_sum = scalar_summary("gen_reg", self.gen_reg)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

        self.d_loss_fake = tf.reduce_mean(self.fake_logits)
        self.d_loss_real = tf.reduce_mean(self.real_logits)
        self.d_loss = self.d_loss_fake - self.d_loss_real
        alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0.0, maxval=1.0)
        dim = self.depth * self.input_height * self.input_width * 3
        vid = tf.reshape(self.real_video, [self.batch_size, dim])
        fake = tf.reshape(self.fake_video, [self.batch_size, dim])
        differences = fake - vid
        x_hat = vid + (alpha * differences)
        d_hat, _ = self.discriminator(tf.reshape(x_hat, [self.batch_size]+self.video_dim), reuse=True) 
        gradients = tf.gradients(d_hat, [x_hat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_pen = tf.reduce_mean((slopes - 1.0)**2)

        self.d_loss = self.d_loss + 10*gradient_pen

        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        #TODO
        d_optim = tf.train.AdamOptimizer(0.0004, beta1=0.0, beta2=0.9) \
                    .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(0.0001, beta1=0.0, beta2=0.9) \
                    .minimize(self.g_loss, var_list=self.g_vars)
        
        tf.global_variables_initializer().run()

        self.g_sum = merge_summary([self.g_loss_pure_sum, self.gen_reg_sum, 
                        self.g_loss_sum, self.d_fake_sum])
        self.d_sum = merge_summary([self.d_loss_sum, self.d_loss_real_sum, 
                        self.d_loss_real_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)
        pass

    
    def discriminator(self, vid, reuse=False):
        """
        input:
            vid: size [N, 32, 64, 64, 3]
        output:
            probabilty: [N, 1]
            without sigmoid: [N, 1]
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            d1 = conv3d(vid, 64, name="d_conv3d1") #[N, 16, 32, 32, 64]
            d1 = lrelu(batch_norm(d1, is_training=True, name="d_bn1"))
            d2 = conv3d(d1, 128, name="d_conv3d2") #[N, 8, 16, 16, 128]
            d2 = lrelu(batch_norm(d2, is_training=True, name="d_bn2"))
            d3 = conv3d(d2, 256, name="d_conv3d3") #[N, 4, 8, 8, 256]
            d3 = lrelu(batch_norm(d3, is_training=True, name="d_bn3"))
            d4 = conv3d(d3, 512, name="d_conv3d4") #[N, 2, 4, 4, 512]
            d4 = lrelu(batch_norm(d4, is_training=True, name="d_bn4"))
            #FC layer here may cause problems.
            d5 = linear(tf.reshape(d4, [self.batch_size, -1]), 1, name="d_lin") #[N, 1]

            return tf.nn.sigmoid(d5), d5
            
    def generator(self, frames):
        """
        input:
            frames: [N, 64, 64, 3]
        output:
            generated frames(32): [N, 32, 64, 64, 3]
        """
        with tf.variable_scope("generator") as scope:
            #Encoder
            e1 = conv2d(frames, 128, name="e_conv2d1") #[N, 32, 32, 128]
            e1 = tf.nn.relu(batch_norm(e1, is_training=True, name="e_bn1"))
            e2 = conv2d(e1, 256, name="e_conv2d2") #[N, 16, 16, 256]
            e2 = tf.nn.relu(batch_norm(e2, is_training=True, name="e_bn2"))
            e3 = conv2d(e2, 512, name="e_conv2d3") #[N, 8, 8, 512]
            e3 = tf.nn.relu(batch_norm(e3, is_training=True, name="e_bn3"))
            e4 = conv2d(e3, 1024, name="e_conv2d4") #[N, 4, 4, 1024]
            e4 = tf.nn.relu(batch_norm(e4, is_training=True, name="e_bn4"))

            #Generator
            pass

    
    def predictor(self, frames):
        #TODO
        pass
    
    @property
    def model_dir(self):
        pass
    
    def save(self, checkpoint_dir, step):
        pass
    
    def load(self, checkpoint_dir):
        pass
    

    

    