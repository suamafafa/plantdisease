#ふつうの
#plantvillage内で分割

import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import os
import time
import glob
import math
import argparse
import sys
import random
import cv2

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
#data args
parser.add_argument("--load_model", action='store_true', help="test is do --load_model")
parser.add_argument("--load_model_path", default=None, help="path for checkpoint")
parser.add_argument("--input_file_path", help="input train data path")
parser.add_argument("--save_dir", help="path for save the model and logs") 
#train args
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epoch", type=int, help="epoch")
parser.add_argument("--print_loss_freq", type=int, default=500, help="print loss epoch frequency")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout_rate. test: 0.0, train=0.2")
parser.add_argument("--nclass", type=int)
parser.add_argument("--model", help="inception, resnet")
parser.add_argument("--gpu_config", default=0, help="0:gpu0, 1:gpu1, -1:both")

a = parser.parse_args()
for k, v in a._get_kwargs():
    print(k, "=", v)

import tensorflow_hub as hub
if a.model == "inception":
	model_size = 299
	module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1", trainable=False)
elif a.model == "resnet":
	model_size = 224
	module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1", trainable=False)

#------paramas------#
sample_size = 162915
#sample_size = 100
#n_classes = 38
#n_classes = 21
n_classes = a.nclass
iteration_num = int(sample_size/a.batch_size*a.epoch)
seed = 1145141919

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
if a.gpu_config == '0':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
elif a.gpu_config == '1':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'))

start_time = time.time()
print("start time : " + str(start_time))

def Resize(imgs): #resize only
	r = imgs
	r = tf.image.resize_images(r, [model_size, model_size])
	return r

with tf.name_scope('LoadImage'):	
	#csv_name = "/home/zhaoyin-t/plant_disease/traindata_int_small_random_disease.csv"
	csv_name = "/home/zhaoyin-t/plant_disease/traindata_seg_int_train.csv"
	filename_queue = tf.train.string_input_producer([csv_name], shuffle=True)
	reader = tf.TextLineReader()
	_, val = reader.read(filename_queue)
	#record_defaults = [["a"],["a"], [0], [0]]
	record_defaults = [["a"],["a"],[0],["a"],[0],[0]]
	#path, _, label, _ = tf.decode_csv(val, record_defaults=record_defaults)
	path, _, _, _, label, _ = tf.decode_csv(val, record_defaults=record_defaults)
	readfile = tf.read_file(path)
	image = tf.image.decode_jpeg(readfile, channels=3)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image = tf.cast(image, dtype=tf.float32)
	image = tf.image.resize_images(image, (model_size, model_size))
	h, w, ch = image.get_shape()
	print(image.get_shape())
	# transform params
	CROP_SIZE = int(h)
	SCALE_SIZE = int(h+20)
	rot90_times = tf.random_uniform([1], 0,5,dtype=tf.int32)[0]
	crop_offset = tf.cast(tf.floor(tf.random_uniform([2], 0, SCALE_SIZE - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
	def transform(img, rot90_times, crop_offset,scale_size=SCALE_SIZE,crop_size=CROP_SIZE):
		with tf.name_scope('transform'):
			r = img
			# rotation
			r = tf.image.rot90(r, k=rot90_times)
			# random crop
			r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
			r = tf.image.crop_to_bounding_box(r, crop_offset[0], crop_offset[1], crop_size, crop_size)
			return r	
	with tf.name_scope('transform_images'):
		image = transform(image, rot90_times, crop_offset, scale_size=SCALE_SIZE, crop_size=CROP_SIZE)
	def contrast(img, param1, param2):
		with tf.name_scope("contrast"):
			r = img
			r = tf.image.random_brightness(r,param1) #明るさ調整
			r = tf.image.random_contrast(r,lower=param2, upper=1/param2) #コントラスト調整	
			return r	
	with tf.name_scope("contrast_images"):
		image = contrast(image, param1=0.1, param2=0.9) 
	label = tf.one_hot(label, depth=n_classes)
	label = tf.cast(label, dtype=tf.float32)
	label_batch, x_batch = tf.train.batch([label, image],batch_size=a.batch_size, allow_smaller_final_batch=False)
	
	"""
	#validation
	fuga  = load_image(csv_name="/home/zhaoyin-t/plant_disease/validationdata.csv", record_defaults = [["a"], ["a"], [0]])
	val_images = fuga[0]
	val_labels = fuga[1]
	"""	
	#test
	#test_csv_name = "/home/zhaoyin-t/plant_disease/testdata_int_disease.csv"	
	test_csv_name = "/home/zhaoyin-t/plant_disease/traindata_seg_int_test.csv"
	test_filename_queue = tf.train.string_input_producer([test_csv_name], shuffle=True)
	test_reader = tf.TextLineReader()
	_, test_val = test_reader.read(test_filename_queue)
	#test_record_defaults = [["a"], ["a"], ["a"], [0], [0]]
	test_record_defaults = [["a"], ["a"], [0], ["a"], [0], [0]]
	#_, test_path, _, test_label, _ = tf.decode_csv(test_val, record_defaults=test_record_defaults)
	test_path, _, _, _, test_label, _ = tf.decode_csv(test_val, record_defaults=test_record_defaults)
	test_readfile = tf.read_file(test_path)
	test_image = tf.image.decode_jpeg(test_readfile, channels=3)
	test_image = tf.image.convert_image_dtype(test_image, dtype=tf.float32)
	test_image = tf.cast(test_image, dtype=np.float32)
	test_image = tf.image.resize_images(test_image, (model_size, model_size))
	test_label = tf.one_hot(test_label, depth=n_classes)
	test_label_batch, test_x_batch = tf.train.batch([test_label, test_image],batch_size=a.batch_size, allow_smaller_final_batch=False)

#---------------Model--#---------------#
#data = tf.placeholder(tf.float32, [None, model_size, model_size, 3])
#label = tf.placeholder(tf.float32, [None, n_classes])
#dropout = tf.placeholder(tf.float32)

am_testing = tf.placeholder(dtype=bool,shape=())
data = tf.cond(am_testing, lambda:test_x_batch, lambda:x_batch)
label = tf.cond(am_testing, lambda:test_label_batch, lambda:label_batch)
drop = tf.placeholder(tf.float32)

def model(data):
	outputs = module(data)
	with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
		#2set: （入力）2048（出力）1000
		#3set:（入力）1000（出力）クラス数	
		logits_ = tf.layers.dense(inputs=outputs, units=1000, activation=tf.nn.leaky_relu, name="dense")
		dropout_ = tf.layers.dropout(inputs=logits_, rate=drop)
		logits = tf.layers.dense(inputs=dropout_, units=n_classes, name="output") 
		return logits

with tf.name_scope("train_logits"):
	logits = model(data)

#--------------Loss&Opt-----------------#
with tf.name_scope("cost"):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))

with tf.name_scope("opt"): 
	#trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "trainable_section")
	trainable_vars = [var for var in tf.trainable_variables()]
	adam = tf.train.AdamOptimizer(0.0002,0.5)
	gradients_vars = adam.compute_gradients(cost, var_list=trainable_vars)	
	train_op = adam.apply_gradients(gradients_vars)

def Accuracy(logits, label):
	correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(label,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return accuracy

with tf.name_scope("accuracy"):
	accuracy = Accuracy(logits, label)

#--------------Summary-----------------#
with tf.name_scope('summary'):
	with tf.name_scope('image_summary'):
		tf.summary.image('image', tf.image.convert_image_dtype(data, dtype=tf.uint8, saturate=True), collections=['train'])
	
	with tf.name_scope("train_summary"):
		cost_summary_train = tf.summary.scalar('train_loss', cost, collections=['train'])
		acc_summary_train = tf.summary.scalar("train_accuracy", accuracy, collections=['train'])
	
	for var in tf.trainable_variables():
		var_summary = tf.summary.histogram(var.op.name + '/Variable_histogram', var, collections=['train'])
	
	for grad, var in gradients_vars:
		grad_summary = tf.summary.histogram(var.op.name + '/Gradients', grad, collections=['train'])


#---------------Session-----------------#
init = tf.global_variables_initializer()
#saver = tf.train.Saver()
tmp_config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="1",
		allow_growth=True
    )
)

saver = tf.train.Saver()
with tf.Session(config=tmp_config) as sess:
	os.mkdir(os.path.join(a.save_dir,'summary'))
	os.mkdir(os.path.join(a.save_dir,'model'))
	sess.run(init)
	print(trainable_vars)
	print("Session Start")
	print("")
	merged = tf.summary.merge_all(key="train")
	summary_writer = tf.summary.FileWriter(os.path.join(a.save_dir,'summary'), graph=sess.graph)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
	for step in range(iteration_num):
		sess.run(train_op, feed_dict={am_testing: False, drop: a.dropout})
		if step % a.print_loss_freq == 0:
			print(step)
			sess.run(cost, feed_dict={am_testing: False})
			train_acc = sess.run(accuracy, feed_dict={am_testing: False, drop: 0.0})
			print("train accuracy", train_acc)
			summary_writer.add_summary(sess.run(merged, feed_dict={am_testing: False, drop: 0.0}), step)
			
			#test
			test_acc = 0
			step_num = -(-186//a.batch_size)
			for i in range(step_num):
				print(i)
				tmp_acc = sess.run(accuracy, feed_dict={am_testing: True, drop: 0.0})
				print(tmp_acc)
				test_acc += tmp_acc
			test_acc = test_acc / step_num
			summary_writer.add_summary(tf.Summary(value=[
				tf.Summary.Value(tag="test_summary/test_accuracy", simple_value=test_acc)
			]), step)
			print("test accuracy", test_acc)
			print()

		if step % (iteration_num/6) == 0:
           	# SAVE
			saver.save(sess, a.save_dir + "/model/model.ckpt")
		
	saver.save(sess, a.save_dir + "/model/model.ckpt")
	print('saved at '+ a.save_dir)
	coord.request_stop()
	coord.join(threads)
end_time = time.time()
print( 'time : ' + str(end_time - start_time))
