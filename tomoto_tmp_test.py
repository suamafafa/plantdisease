#tomato only

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
from keras import backend as K
from skimage.transform import resize

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", action='store_true', help="test is do --load_model")
parser.add_argument("--load_model_path", default=None, help="path for checkpoint")
parser.add_argument("--augm", action='store_true', help="augmentation is do")
parser.add_argument("--val", action='store_true', help="validation is do")
parser.add_argument("--save_dir", help="path for save the model and logs") 
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epoch", type=int, help="epoch")
parser.add_argument("--print_loss_freq", type=int, default=500, help="print loss epoch frequency")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout_rate. test: 0.0, train=0.2")
parser.add_argument("--nclass", type=int)
parser.add_argument("--model", help="inception, resnet")
parser.add_argument("--gpu_config", default=0, help="0:gpu0, 1:gpu1, -1:both")
parser.add_argument("--gpu", default=1)

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
sample_size = 17257 #17257:tomato_df_train, 18160:tomato_df
n_classes = 10
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
	#csv_name = "/home/zhaoyin-t/plant_disease/traindata_seg_int_train_random.csv"
	csv_name = "/home/zhaoyin-t/plant_disease/tomato_df_train_random.csv"
	filename_queue = tf.train.string_input_producer([csv_name], shuffle=True, num_epochs=None)
	reader = tf.TextLineReader()
	_, val = reader.read(filename_queue)
	record_defaults = [["a"], ["a"], [0], ["a"], [0], [0]]
	#record_defaults = [["a"],["a"], [0], [0]]
	path, _, _, _, label, _ = tf.decode_csv(val, record_defaults=record_defaults)
	#path, _, _, label = tf.decode_csv(val, record_defaults=record_defaults)
	readfile = tf.read_file(path)
	image = tf.image.decode_jpeg(readfile, channels=3)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image = tf.cast(image, dtype=tf.float32)
	image = tf.image.resize_images(image, (model_size, model_size))
	
	if a.augm is True:
	
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

	t est_csv_name = "/home/zhaoyin-t/plant_disease/tomato_test_only_tomato.csv"
	#test_csv_name = "/home/zhaoyin-t/plant_disease/nontomato_test_rm_h.csv"
	test_filename_queue = tf.train.string_input_producer([test_csv_name], shuffle=True)
	test_reader = tf.TextLineReader()
	_, test_val = test_reader.read(test_filename_queue)
	#record_defaults = [["a"], ["a"], ["a"], [0], ["a"], [0]]
	record_defaults = [["a"], ["a"], [0], ["a"], [0], [0]]
	#_, test_path, _, _, _, test_label= tf.decode_csv(test_val, record_defaults=record_defaults)
	test_path, _, _, _, test_label, _ = tf.decode_csv(test_val, record_defaults=record_defaults)
	test_readfile = tf.read_file(test_path)
	test_image = tf.image.decode_jpeg(test_readfile, channels=3)
	test_image = tf.image.convert_image_dtype(test_image, dtype=tf.float32)
	test_image = tf.cast(test_image, dtype=np.float32)
	test_image = tf.image.resize_images(test_image, (model_size, model_size))
	test_label = tf.one_hot(test_label, depth=n_classes)
	test_label_batch, test_x_batch = tf.train.batch([test_label, test_image],batch_size=84)

#---------------Model--#---------------#
#data = tf.placeholder(tf.float32, [None, model_size, model_size, 3])
#label = tf.placeholder(tf.float32, [None, n_classes])
#dropout = tf.placeholder(tf.float32)

am_training = tf.placeholder(dtype=bool,shape=())
#img_pl = tf.placeholder(tf.float32, [None, model_size, model_size, 3])
#label_pl = tf.placeholder(tf.float32, [None, n_classes])
data = tf.cond(am_training, lambda:x_batch, lambda:test_x_batch)
#data = x_batch
label = tf.cond(am_training, lambda:label_batch, lambda:test_label_batch)
#label = label_batch
drop = tf.placeholder(tf.float32)

tmp_img = np.ndarray(shape=(1,model_size,model_size,3), dtype=float)
tmp_label = np.ndarray(shape=(1,n_classes), dtype=float)

with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
	def model(data):
		outputs = module(data)
		#2set: （入力）2048（出力）1000
		#3set:（入力）1000（出力）クラス数
		logits_ = tf.layers.dense(inputs=outputs, units=n_classes, name="dence")
		dropout_ = tf.layers.dropout(inputs=logits_, rate=drop)
		y = tf.layers.dense(inputs=dropout_, units=n_classes, name="model_output") 
		return y
	#with tf.name_scope("train_logits"):
	logits = model(data)
	logits = tf.identity(logits, name="output")

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

def grad_cam(x, sess, pred_class, nclass, size):
	print("Setting gradients to 1 for target class and rest to 0")
	one_hot = tf.sparse_to_dense(pred_class, [nclass], 1.0)
	#logits = tf.get_default_graph().get_tensor_by_name("train_logits/output:0")
	logits = sess.graph.get_tensor_by_name("train_logits/output:0")
	print("logits", logits)
	signal = tf.multiply(logits, one_hot) #logits = model(output)
	loss = tf.reduce_mean(signal)
	print("loss", loss)
	reduced_loss = tf.reduce_sum(signal[0])
	print(reduced_loss)
	
	#conv_layer = tf.get_default_graph().get_tensor_by_name("train_logits/module_apply_default/resnet_v2_50/block4/unit_3/bottleneck_v2/add:0")
	conv_layer = sess.graph.get_tensor_by_name("train_logits/module_apply_default/resnet_v2_50/block4/unit_3/bottleneck_v2/add:0")
	print("conv", conv_layer)

	grads = tf.gradients(loss,conv_layer)[0]
	print("grads", grads)
    # Normalizing the gradients
	norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

	output, grads_val = sess.run([conv_layer, norm_grads])
	output = output[0]
	print("output", output.shape)
	grads_val = grads_val[0]
	print("gradval", grads_val.shape)
	weights = np.mean(grads_val, axis = (0, 1))             # [512]	
	cam = np.ones(output.shape[0 : 2], dtype = np.float32)  # [7,7]

    # Taking a weighted averagei
	for i, w in enumerate(weights):
		cam += w * output[:,:,i]

    # Passing through ReLU
	cam = np.maximum(cam, 0)
	cam = cam / np.max(cam)
	cam = resize(cam, (size,size))

    # Converting grayscale to 3-D
	jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
	return jetcam

#---------------Session-----------------#
init = tf.global_variables_initializer()
#saver = tf.train.Saver()
tmp_config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0",
		allow_growth = True

    )
)

saver = tf.train.Saver()
with tf.Session(config=tmp_config) as sess:
	if a.load_model is not True:
		if not os.path.exists(a.save_dir):
			os.mkdir(a.save_dir)

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
		graph = tf.get_default_graph()
		placeholders = [ op for op in graph.get_operations() if op.type == "Placeholder"]
		print("placeholder", placeholders)
		for step in range(iteration_num):
			sess.run(train_op, feed_dict={am_training: True ,drop:a.dropout})
			if step % a.print_loss_freq == 0:
				print(step)
				train_acc = sess.run(accuracy, feed_dict={am_training: True, drop: 0.0})
				print("train accuracy", train_acc)
				summary_writer.add_summary(sess.run(merged, feed_dict={am_training: True, drop: 0.0}), step)
		
				#test
				test_acc = 0
				test_num = pd.read_csv(test_csv_name, header=None).shape[0]
				step_num = -(-test_num//a.batch_size)
				print("step_num", step_num)
				for i in range(step_num):
					print(i)
					tmp_acc = sess.run(accuracy, feed_dict={am_training: False, drop: 0.0})
					print(tmp_acc)
					test_acc += tmp_acc
				test_acc = test_acc / step_num
				summary_writer.add_summary(tf.Summary(value=[
				tf.Summary.Value(tag="test_summary/test_accuracy", simple_value=test_acc)]), step)
				print("test accuracy", test_acc)
				print()
				"""	
				tmp_acc = 0
				csv = pd.read_csv("tomato_test.csv", header=None)
				num = csv.shape[0]
				for i in range(num):
					print(csv.iloc[i])
					img = cv2.imread(csv.iloc[i,1])	
					print(img.shape)
					img = img//255.0
					img = np.resize(img, [1, model_size, model_size, 3])
					print(img.shape)
					label = csv.iloc[i,5]
					#logits = sess.graph.get_tensor_by_name("model/output:0")
					#logits = sess.run(logits, feed_dict={am_testing:True, test_img:img, test_label:label, drop:0.0})
					print(logits.shape)
					lo = model(img)
					tmp_acc += sess.run(Accuracy(lo, label))
					print(tmp_acc)
				test_acc2 = test_acc / num
				summary_writer.add_summary(tf.Summary(value=[
				tf.Summary.Value(tag="test_summary/test2_accuracy", simple_value=test_acc2)]), step)
				print("test2 accuracy", test_acc2)
				print()
				"""
			if step % (iteration_num/5) == 0:
        	   	# SAVE
				saver.save(sess, a.save_dir + "/model/model.ckpt")
		
		saver.save(sess, a.save_dir + "/model/model.ckpt")
		print('saved at '+ a.save_dir)
		
		coord.request_stop()
		coord.join(threads)
	else:
		ckpt  = tf.train.get_checkpoint_state(a.load_model_path + "/model")
		last_model = ckpt.model_checkpoint_path
		saver.restore(sess, last_model)
		print("load" + last_model)
		tmp_acc = 0
		csv = pd.read_csv("tomato_test.csv", header=None)
		num = csv.shape[0]
		#step_num = -(-num//a.batch_size)
		for i in range(num):
			print(csv.iloc[i,1])
			img_bgr = cv2.imread(csv.iloc[i,1])
			img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
			img = img//255.0
			img = np.resize(img, [1, model_size, model_size, 3])
			print(img.shape)
			label = csv.iloc[i,5]
			logits = sess.graph.get_tensor_by_name("model/output:0")
			logits = sess.run(logits, feed_dict={am_testing:True, test_img:img, test_label:label, drop:0.0})
			print(logits.shape)
			#lo = model(img)
			tmp_acc += Accuracy(lo, label)
			print(tmp_acc)
		test_acc = test_acc / num
		#summary_writer.add_summary(tf.Summary(value=[
		#tf.Summary.Value(tag="test_summary/test_accuracy", simple_value=test_acc)]), step)
		print("test accuracy", test_acc)
		print()
end_time = time.time()
print( 'time : ' + str(end_time - start_time))
