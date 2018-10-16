import tensorflow as tf
import numpy as np
import datetime
import os 
import time
import glob
import math 
import argparse
import sys
sys.path.append("/home/zhaoyin-t/function")
from Plot_data_on_image import *
#from __future__ import division, print_function, absolute_import

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("-print_loss_freq", default=1000)
parser.add_argument("--save_dir")
parser.add_argument("--epoch", type=int, default=500)
parser.add_argument("--gpu_config", default=-1)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

a = parser.parse_args()
for k, v in a._get_kwargs():
	print(k, "=", v)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
if a.gpu_config == '0':
	config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True,visible_device_list='0'))
elif a.gpu_config == '1':
	config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True,visible_device_list='1'))

start_time = time.time()
print("start time : " + str(start_time))

sample_size = 100
image_size = 28
#image_size = 256
#image_dim = image_size*image_size
image_dim = 784
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100 # Noise data points

# Generator
#Input:Noise
#Output:Image
def generator(x, reuse=False):
	x = tf.layers.dense(x, units=6 * 6 * 128)
	x = tf.nn.tanh(x)
    # Reshape to a 4-D array of images: (batch, height, width, channels)
	# New shape: (batch, 6, 6, 128) #変更するかも
	x = tf.reshape(x, shape=[-1, 6, 6, 128])
	# Deconvolution, image shape: (batch, 14, 14, 64)
	x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
	# Deconvolution, image shape: (batch, 28, 28, 1)
	x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
	# Apply sigmoid to clip values between 0 and 1
	x = tf.nn.sigmoid(x)
	return x

# Discriminator
#Input:Image
#Output:Prediction 
def discriminator(x, reuse=False):
	# Typical convolutional neural network to classify images.
	x = tf.layers.conv2d(x, 64, 5)
	x = tf.nn.tanh(x)
	x = tf.layers.average_pooling2d(x, 2, 2)
	x = tf.layers.conv2d(x, 128, 5)
	x = tf.nn.tanh(x)
	x = tf.layers.average_pooling2d(x, 2, 2)
	x = tf.contrib.layers.flatten(x)
	x = tf.layers.dense(x, 1024)
	x = tf.nn.tanh(x)
    # Output 2 classes: Real and Fake images
	x = tf.layers.dense(x, 2)
	return x

#--------------Load Image-----------------#
with tf.name_scope("LoadImage"):
	csv_name = "/home/zhaoyin-t/plant_disease/test2.csv"
	filename_queue = tf.train.string_input_producer([csv_name], shuffle=True)
	reader = tf.TextLineReader()
	key, val = reader.read(filename_queue)
	val_defaults = [["aa"], [1], [1], [1]]
	path, label, label_disease, label_plant = tf.decode_csv(val, record_defaults=val_defaults)	
	readfile = tf.read_file(path)
	image = tf.image.decode_jpeg(readfile, channels=3)
	image.set_shape([256,256,3])
	image = tf.image.resize_images(image, [image_size, image_size])
	image = tf.cast(image, dtype=np.float32)
	image = image/255.0	
	bel_batch, x_batch = tf.train.batch([label, image],batch_size=a.batch_size)
	iteration_num = int(sample_size/a.batch_size*a.epoch)

#---------------Model-----------------#
#Network Input
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
real_image_input = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name='disc_input')

#Build Gen Network
gen_sample = generator(noise_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)
disc_concat = tf.concat([disc_real, disc_fake], axis=0)

#stacked=積みあげた
stacked_gan = discriminator(gen_sample, reuse=True)

# Build Targets (real or fake images)
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

#---------------Loss&Opt-----------------#
# Build Loss
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_concat, labels=disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=gen_target))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer()
optimizer_disc = tf.train.AdamOptimizer()

# Generator Network Variables
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

#---------------Summary-----------------#


#---------------Session-----------------#
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	print("Session Start")
	for step in range(iteration_num):
		if not os.path.exists(a.save_dir):
			os.mkdir(os.path.join(a.save_dir, "model"))

		#run_x = sess.run(x_batch)
		run_x = mnist.train.next_batch(a.batch_size)
		#The first falf of data are real image=1, the other falf is fake=0
		batch_disc_y = np.concatenate(
            [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
        # Generator tries to fool the discriminator, thus targets are 1.
		batch_gen_y = np.ones([batch_size])
		
		#Generate noise to feed to the generator
		z = np.random.uniform(-1., 1., size=[a.batch_size, noise_dim])

		feed_dict = {real_image_input: run_x, noise_input: z,
                     disc_target: batch_disc_y, gen_target: batch_gen_y}
		tg, td, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss], feed_dict=feed_dict)

		if step % a.print_loss_freq == 0 or step == 1:
			print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (step, gl, dl))
			saver.save(sess, a.save_dir + "/model/model.ckpt")
	
	saver.save(sess, a.save_dir + "/model/model.ckpt")
	print('saved at '+ a.save_dir)
	coord.request_stop()
	coord.join(threads)
end_time = time.time()
print( 'time : ' + str(end_time - start_time))
