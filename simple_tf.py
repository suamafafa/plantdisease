#transfer learning 
#without lightGBM

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

parser = argparse.ArgumentParser()
#data args
parser.add_argument("--load_model", action='store_true', help="test is do --load_model")
parser.add_argument("--load_model_path", default=None, help="path for checkpoint")
parser.add_argument("--input_file_path", help="input train data path")
parser.add_argument("--save_dir", help="path for save the model and logs")
#train args
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epoch", type=int, default=500, help="epoch")
parser.add_argument("--print_loss_freq", type=int, default=500, help="print loss epoch frequency")
parser.add_argument("--gpu_config", default=-1, help="0:gpu, 1:gpu1, -1:both")

a = parser.parse_args()
for k, v in a._get_kwargs():
    print(k, "=", v)

#------paramas------#
sample_size = 1629150


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
if a.gpu_config == '0':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
elif a.gpu_config == '1':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'))

start_time = time.time()
print("start time : " + str(start_time))

with tf.name_scope('LoadImage'):
	csv_name = "/home/zhaoyin-t/plant_disease/traindata.csv"
	if a.load_model is not True:
		filename_queue = tf.train.string_input_producer([csv_name], shuffle=True)
	else:
		filename_queue = tf.train.string_input_producer(input_paths, num_epochs=1, shuffle=False)	
	reader = tf.TextLineReader()
	_, val = reader.read(filename_queue)
	record_defaults = [["a"], ["a"]]
	path, label = tf.decode_csv(val, record_defaults=record_defaults)
	readfile = tf.read_file(path)
	image = tf.image.decode_jpeg(readfile, channels=3)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image.set_shape([256,256,3])	
	image = image/255.0
	label_batch, x_batch = tf.train.batch([label, image],batch_size=32, allow_smaller_final_batch=True)
	iteration_num = int(sample_size/a.batch_size*a.epoch)

	parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

#---------------Session-----------------#
init = tf.global_variables_initializer()
#saver = tf.train.Saver()
with tf.Session(config=config) as sess:
	if a.load_model is not True:
		sess.run(init)
		print("Session Start")
		print("")
		#print('total parameters : ',sess.run(parameter_count))
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		for step in range(iteration_num):
			#print(sess.run(label_batch))
		

		coord.request_stop()
		coord.join(threads)

end_time = time.time()
print( 'time : ' + str(end_time - start_time))
