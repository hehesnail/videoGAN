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
                input_fname_pattern="*.jpeg",checkpoint_dir=None, sample_dir=None, critic=5):
        self.sess = sess

        self.depth = depth
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.z_dim = z_dim
        self.lam = lam
        self.video_dim = [self.depth, self.output_height, self.output_width, 3]

        self.batch_size = batch_size
        self.dataset = dataset
        self.critic = critic

        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        self.build_model()

    def build_model(self):
        self.frames = tf.placeholder(tf.float32, [self.batch_size, self.output_height, self.output_width, 3]) #[N,H,W,3]
        self.real_video = tf.placeholder(tf.float32, [self.batch_size] + self.video_dim) #[N,D,H,W,3]

        self.fake_video, self.foreground, self.background, self.mask, self.gen_reg = self.generator(self.frames)
        self.sample_video = self.predictor(self.frames)
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
        dim = self.depth * self.output_height * self.output_width * 3

        real = tf.reshape(self.real_video, [self.batch_size, dim])
        fake = tf.reshape(self.fake_video, [self.batch_size, dim])

        differences = fake - real
        x_hat = fake + (alpha * differences)
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
        d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5, beta2=0.9) \
                    .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5, beta2=0.9) \
                    .minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.g_sum = merge_summary([self.g_loss_pure_sum, self.gen_reg_sum,
                        self.g_loss_sum, self.d_fake_sum])
        self.d_sum = merge_summary([self.d_loss_sum, self.d_real_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print("[*] Load SUCCESS")
        else:
            print("[!] Load failed...")

        for epoch in range(config.epoch):
            self.data = load_data(self.dataset, self.input_fname_pattern)
            print(len(self.data))
            batch_idxs = len(self.data) // config.batch_size
            for idx in range(0, batch_idxs-1):
                batch_files = self.data[idx*config.batch_size : (idx+1)*config.batch_size]
                video_files = self.data[idx*config.batch_size : (idx+1)*config.batch_size+32]
                batch = [get_image(batch_file, resize_h=self.output_height,
                                resize_w=self.output_width) for batch_file in batch_files]
                #batch images: [N, 64, 64 ,3]
                batch_images = np.array(batch).astype(np.float32)
                #batch videos: [N, 32, 64, 64, 3]
                batch_videos = np.zeros((self.batch_size, self.depth, self.output_height, self.output_width, 3))
                for i in range(self.batch_size):
                    for j in range(self.depth):
                        batch_videos[i, j, :, :, :] = get_image(video_files[i+j], resize_h=self.output_height,
                                                               resize_w=self.output_width)
                #batch_videos = read_and_process_video(batch_files,self.batch_size,self.depth)
                #update the discriminator
                for i in range(self.critic):
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={
                            self.frames: batch_images,
                            self.real_video: batch_videos
                        })
                    self.writer.add_summary(summary_str, counter)

                #update the G
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={
                            self.frames:batch_images
                        })
                self.writer.add_summary(summary_str, counter)
                errD = self.d_loss.eval({self.frames:batch_images, self.real_video:batch_videos})
                errG = self.g_loss.eval({self.frames:batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs, time.time()-start_time, errD, errG))
                if np.mod(counter, 100) == 1:
                    predicted_frames, d_loss, g_loss = self.sess.run(
                        [self.predictor, self.d_loss, self.g_loss], feed_dict={self.frames:batch_images})
                    #predicted_frames [N, 32, 64, 64, 3]
                    process_and_write_video(predicted_frames, "videos"+counter, config.sample_dir)
                    print("Writing videos...")
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                if np.mod(counter, 1000) == 1:
                    self.save(config.checkpoint_dir, counter)


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
            d1 = lrelu(layer_norm(d1, is_training=True, name="d_ln1"))
            d2 = conv3d(d1, 128, name="d_conv3d2") #[N, 8, 16, 16, 128]
            d2 = lrelu(layer_norm(d2, is_training=True, name="d_ln2"))
            d3 = conv3d(d2, 256, name="d_conv3d3") #[N, 4, 8, 8, 256]
            d3 = lrelu(layer_norm(d3, is_training=True, name="d_ln3"))
            d4 = conv3d(d3, 512, name="d_conv3d4") #[N, 2, 4, 4, 512]
            d4 = lrelu(layer_norm(d4, is_training=True, name="d_ln4"))
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
            e1 = tf.nn.relu(e1)
            e2 = conv2d(e1, 256, name="e_conv2d2") #[N, 16, 16, 256]
            e2 = tf.nn.relu(batch_norm(e2, is_training=True, epsilon=1e-3, name="e_bn1"))
            e3 = conv2d(e2, 512, name="e_conv2d3") #[N, 8, 8, 512]
            e3 = tf.nn.relu(batch_norm(e3, is_training=True, epsilon=1e-3, name="e_bn2"))
            e4 = conv2d(e3, 1024, name="e_conv2d4") #[N, 4, 4, 1024]
            e4 = tf.nn.relu(batch_norm(e4, is_training=True, epsilon=1e-3, name="e_bn3"))

            #Generator
            #Forground stream
            z = tf.reshape(e4, shape=[self.batch_size, 2, 4, 4, 512]) #[N, 2, 4, 4, 512]
            gf1 = deconv3d(z, [self.batch_size, 4, 8, 8, 256], name="gf_deconv3d1") #[N, 4, 8, 8, 256]
            gf1 = tf.nn.relu(batch_norm(gf1, is_training=True, name="gf_bn1"))
            gf2 = deconv3d(gf1, [self.batch_size, 8, 16, 16, 128], name="gf_deconv3d2") #[N, 8, 16, 16, 128]
            gf2 = tf.nn.relu(batch_norm(gf2, is_training=True, name="gf_bn2"))
            gf3 = deconv3d(gf2, [self.batch_size, 16, 32, 32, 64], name="gf_deconv3d3") #[N, 16, 32, 32, 64]
            gf3 = tf.nn.relu(batch_norm(gf3, is_training=True, name="gf_bn3"))

            #Foreground video
            foreground = deconv3d(gf3, [self.batch_size, 32, 64, 64, 3], name="f_deconv3d") #[N, 32, 64, 64, 3]
            foreground = tf.tanh(foreground)

            #mask
            mask = deconv3d(gf3, [self.batch_size, 32, 64, 64, 1], name="m_deconv3d") #[N, 32, 64, 64, 1]
            mask = tf.sigmoid(mask)

            #Background
            gb1 = deconv2d(e4, [self.batch_size, 8, 8, 512], name="gb_deconv2d1") #[N,8,8,512]
            gb1 = tf.nn.relu(batch_norm(gb1, is_training=True, name="gb_bn1"))
            gb2 = deconv2d(gb1, [self.batch_size, 16, 16, 256], name="gb_deconv2d2") #[N, 16, 16, 256]
            gb2 = tf.nn.relu(batch_norm(gb2, is_training=True, name="gb_bn2"))
            gb3 = deconv2d(gb2, [self.batch_size, 32, 32, 128], name="gb_deconv2d3") #[N, 32, 32, 128]
            gb3 = tf.nn.relu(batch_norm(gb3, is_training=True, name="g_bn3"))
            background = deconv2d(gb3, [self.batch_size, 64, 64, 3], name="gb_deconv2d4") #[N, 64, 64, 3]

            background = tf.tanh(background)
            background = tf.expand_dims(background, axis=1) #[N, 1, 64, 64, 3]

            mask_rep = tf.tile(mask, [1, 1, 1, 1, 3]) #[N, 32, 64, 64, 3]
            background_rep = tf.tile(background, [1, 32, 1, 1, 1]) #[N, 32, 64, 64, 3]

            video = mask_rep*foreground + (1-mask_rep)*background_rep #[N, 32, 64, 64, 3]
            gen_reg = tf.reduce_mean(tf.square(frames - video[:, 0, :, :, :]))

            return video, foreground, background, mask, gen_reg

    def predictor(self, frames):
        """
        reuse the variables of the generator to predict the future frames
        input:
            frames: [N, 64, 64, 3]
        output:
            generated frames(32): [N, 32, 64, 64, 3]
        """
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            #Encoder
            e1 = conv2d(frames, 128, name="e_conv2d1") #[N, 32, 32, 128]
            e1 = tf.nn.relu(e1)
            e2 = conv2d(e1, 256, name="e_conv2d2") #[N, 16, 16, 256]
            e2 = tf.nn.relu(batch_norm(e2, epsilon=1e-3, name="e_bn1"))
            e3 = conv2d(e2, 512, name="e_conv2d3") #[N, 8, 8, 512]
            e3 = tf.nn.relu(batch_norm(e3, epsilon=1e-3, name="e_bn2"))
            e4 = conv2d(e3, 1024, name="e_conv2d4") #[N, 4, 4, 1024]
            e4 = tf.nn.relu(batch_norm(e4, epsilon=1e-3, name="e_bn3"))

            #Generator
            #Forground stream
            z = tf.reshape(e4, shape=[self.batch_size, 2, 4, 4, 512]) #[N, 2, 4, 4, 512]
            gf1 = deconv3d(z, [self.batch_size, 4, 8, 8, 256], name="gf_deconv3d1") #[N, 4, 8, 8, 256]
            gf1 = tf.nn.relu(batch_norm(gf1, name="gf_bn1"))
            gf2 = deconv3d(gf1, [self.batch_size, 8, 16, 16, 128], name="gf_deconv3d2") #[N, 8, 16, 16, 128]
            gf2 = tf.nn.relu(batch_norm(gf2, name="gf_bn2"))
            gf3 = deconv3d(gf2, [self.batch_size, 16, 32, 32, 64], name="gf_deconv3d3") #[N, 16, 32, 32, 64]
            gf3 = tf.nn.relu(batch_norm(gf3, name="gf_bn3"))

            #Foreground video
            foreground = deconv3d(gf3, [self.batch_size, 32, 64, 64, 3], name="f_deconv3d") #[N, 32, 64, 64, 3]
            foreground = tf.tanh(foreground)

            #mask
            mask = deconv3d(gf3, [self.batch_size, 32, 64, 64, 1], name="m_deconv3d") #[N, 32, 64, 64, 1]
            mask = tf.sigmoid(mask)

            #Background
            gb1 = deconv2d(e4, [self.batch_size, 8, 8, 512], name="gb_deconv2d1") #[N,8,8,512]
            gb1 = tf.nn.relu(batch_norm(gb1, name="gb_bn1"))
            gb2 = deconv2d(gb1, [self.batch_size, 16, 16, 256], name="gb_deconv2d2") #[N, 16, 16, 256]
            gb2 = tf.nn.relu(batch_norm(gb2, name="gb_bn2"))
            gb3 = deconv2d(gb2, [self.batch_size, 32, 32, 128], name="gb_deconv2d3") #[N, 32, 32, 128]
            gb3 = tf.nn.relu(batch_norm(gb3, name="g_bn3"))
            background = deconv2d(gb3, [self.batch_size, 64, 64, 3], name="gb_deconv2d4") #[N, 64, 64, 3]

            background = tf.tanh(background)
            background = tf.expand_dims(background, axis=1) #[N, 1, 64, 64, 3]

            mask_rep = tf.tile(mask, [1, 1, 1, 1, 3]) #[N, 32, 64, 64, 3]
            background_rep = tf.tile(background, [1, 32, 1, 1, 1]) #[N, 32, 64, 64, 3]

            video = mask_rep*foreground + (1-mask_rep)*background_rep #[N, 32, 64, 64, 3]

            return video

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.dataset, self.batch_size,
                            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "VGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print("[*] Reading checkpoints")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print("[*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print("[*] Failed to find a checkpoint")
            return False, 0
