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
parser.add_argument("--dropout", type=float, default=0.2, help="dropout_rate. test: 0.0, train=0.2") #!!!!
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
n_classes = 21
iteration_num = int(sample_size/a.batch_size*a.epoch)
seed = 1145141919
Ip = 1
xi = 1e-6
eps = 8.0

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
if a.gpu_config == '0':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
elif a.gpu_config == '1':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'))

start_time = time.time()
print("start time : " + str(start_time))

def Get_normalized_vector(d):
	with tf.name_scope('get_normalized_vec'):
		d /= (1e-12 + tf.reduce_sum(tf.abs(d), axis=[1,2,3], keep_dims=True))
		d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.), axis=[1,2,3], keep_dims=True))	
		return d

def KL_divergence(p, q):
	# KL = 竏ｫp(x)log(p(x)/q(x))dx
	#    = 竏ｫp(x)(log(p(x)) - log(q(x)))dx
	kld = tf.reduce_mean(tf.reduce_sum(p * (tf.log(p + 1e-14) - tf.log(q + 1e-14)), axis=[1]))
	return kld

def Accuracy(logits,label):
	correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(label,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return accuracy

def scale(x):
	myu = tf.reduce_mean(x)
	a = (x-myu)**2
	return tf.sqrt(a)

def Generate_perturbation(x): #画像に加えるノイズの生成
	d = tf.random_normal(shape=tf.shape(x))
	for i in range(Ip):
		d = xi * Get_normalized_vector(d)
		p = model(x)
		q = model(x + d)
		d_kl = KL_divergence(p,q)
		grad = tf.gradients(d_kl, [d], aggregation_method=2)[0] # d(d_kl)/d(d)
		print("grad", grad)
		d = tf.stop_gradient(grad)
		print(d)
		return 0.25 * scale(x) * Get_normalized_vector(d)

def Get_VAT_loss(x,r): #生画像とノイズ画像の出力の差を計算
	with tf.name_scope('Get_VAT_loss'):
		p = tf.stop_gradient(model(x))
		q = model(x + r)
		loss = KL_divergence(p,q)
		return tf.identity(loss, name='vat_loss')

def Affine_loss(x,x_aug): #生画像とノイズ画像の出力の差を計算
	with tf.name_scope('Get_VAT_loss'):
		p = tf.stop_gradient(model(x))
		q = model(x_aug)
		loss = KL_divergence(p,q)
		return tf.identity(loss, name='affine_loss')

def ImageDecorate(imgs, model_size): #with resize
	r = imgs
	print(r.get_shape())
	_, h, w, ch = r.get_shape() 
	CROP_SIZE = int(h)
	SCALE_SIZE = int(h+20)
	seed = 1141919
	rot90_times = tf.random_uniform([1], 0,5,dtype=tf.int32)[0]
	crop_offset = tf.cast(tf.floor(tf.random_uniform([2], 0, SCALE_SIZE - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
	print(crop_offset.get_shape())
	r = tf.image.rot90(r, k=rot90_times)
	r = tf.image.resize_images(r, [SCALE_SIZE, SCALE_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	print(r.get_shape())
	r = tf.image.crop_to_bounding_box(r, crop_offset[0], crop_offset[1], CROP_SIZE, CROP_SIZE)
	print(r.get_shape())
	param1 = 0.1
	param2 = 0.9
	r = tf.image.random_brightness(r,param1) #明るさ調整
	r = tf.image.random_contrast(r,lower=param2, upper=1/param2) #コントラスト調整	#r = tf.image.resize_images(r, [model_size, model_size])
	r = tf.image.resize_images(r, (model_size, model_size), method=tf.image.ResizeMethod.AREA)
	return r

with tf.name_scope('LoadImage'):
	#csv_name = "/home/zhaoyin-t/plant_disease/traindata_int_small_random_disease.csv"
	csv_name = "/home/zhaoyin-t/plant_disease/traindata_seg_int_random.csv"
	filename_queue = tf.train.string_input_producer([csv_name], shuffle=True)
	reader = tf.TextLineReader()
	_, val = reader.read(filename_queue)
	record_defaults = [["a"],["a"],[0],["a"],[0],[0]]
	#record_defaults = [["a"],[0], [0], [0]]
	path, _, _,  _, label, _ = tf.decode_csv(val, record_defaults=record_defaults)
	readfile = tf.read_file(path)
	image = tf.image.decode_jpeg(readfile, channels=3)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image = tf.cast(image, dtype=np.float32)
	
	height,width,ch = image.get_shape()
	# transform params
	CROP_SIZE = 256
	SCALE_SIZE = 286
	rot90_times = tf.random_uniform([1], 0,5,dtype=tf.int32)[0]
	crop_offset = tf.cast(tf.floor(tf.random_uniform([2], 0, SCALE_SIZE - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
	def transform(img, rot90_times, crop_offset,scale_size=286,crop_size=256):
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
		image = contrast(image, param1=0.25, param2=0.75)

	image = tf.image.resize_images(image, (model_size, model_size), method=tf.image.ResizeMethod.AREA)
	label = tf.one_hot(label, depth=n_classes)
	label_batch, x_batch = tf.train.batch([label, image],batch_size=a.batch_size, allow_smaller_final_batch=False)
	label_batch = tf.cast(label_batch, dtype=np.float32)
	
	"""
	#validation
	fuga  = load_image(csv_name="/home/zhaoyin-t/plant_disease/validationdata.csv", record_defaults = [["a"], ["a"], [0]])
	val_images = fuga[0]
	val_labels = fuga[1]
	"""	
	"""
	#unlabel
	unlabel_image_np = np.load("unlabel_image.npy")
	unlabel_images_tf = tf.convert_to_tensor(unlabel_image_np, tf.float32)
	unlabel_images_tf = tf.image.resize_images(unlabel_images_tf, (model_size, model_size), method=tf.image.ResizeMethod.AREA)
	unlabel_image_queue = tf.train.input_producer(unlabel_images_tf, shuffle=False)
	unlabel_image_dequeue = unlabel_image_queue.dequeue()
	unlabel_images = tf.train.batch([unlabel_image_dequeue], batch_size=a.batch_size, allow_smaller_final_batch=True)

	#test
	test_csv_name = "/home/zhaoyin-t/plant_disease/testdata_int_disease.csv"
	test_filename_queue = tf.train.string_input_producer([test_csv_name], shuffle=True)
	test_reader = tf.TextLineReader()
	_, test_val = test_reader.read(test_filename_queue)
	record_defaults = [["a"], ["a"], ["a"], [0], [0]]
	#record_defaults = [["a"],[0], [0], [0]]
	_, test_path, _, test_label, _ = tf.decode_csv(test_val, record_defaults=record_defaults)
	test_readfile = tf.read_file(test_path)
	test_image = tf.image.decode_jpeg(test_readfile, channels=3)
	test_image = tf.image.convert_image_dtype(test_image, dtype=tf.float32)
	test_image = tf.cast(test_image, dtype=np.float32)
	test_image = tf.image.resize_images(test_image, (model_size, model_size))
	test_label = tf.one_hot(test_label, depth=n_classes)
	test_label_batch, test_x_batch = tf.train.batch([test_label, test_image],batch_size=a.batch_size, allow_smaller_final_batch=False)
	"""
#---------------Model--#---------------#
am_testing = tf.placeholder(dtype=bool,shape=())
#img_data = tf.cond(am_testing, lambda:test_x_batch, lambda:x_batch)
#label = tf.cond(am_testing, lambda:test_label_batch, lambda:label_batch)
img_data = x_batch
label = label_batch
drop = tf.placeholder(tf.float32)
#un_img_data = unlabel_images

def model(data): #ソフトマックスまで
	outputs = module(data)
	with tf.variable_scope("NN", reuse=tf.AUTO_REUSE):
		#2set: （入力）2048（出力）1000
		#3set:（入力）1000（出力）クラス数	
		logits_ = tf.layers.dense(inputs=outputs, units=1000, activation=tf.nn.leaky_relu, name="dense")
		dropout_ = tf.layers.dropout(inputs=logits_, rate=drop)
		logits = tf.layers.dense(inputs=dropout_, units=n_classes, name="output") 
		out = tf.nn.softmax(logits)
		return out

with tf.name_scope("Model"):
	with tf.name_scope("model_outputs"):
		data_out = model(img_data)
		#un_data_out = model(un_img_data) #追加
	with tf.name_scope('Generate_perturbation'):
		# generate perturbation
		r_adv = Generate_perturbation(img_data)
		#un_r_adv = Generate_perturbation(un_img_data)
	    # add perturbation onto x
		data_r = img_data + r_adv
		#un_data_r = un_img_data + un_r_adv
		#img_data_aug = ImageDecorate(img_data, model_size=model_size)
	
#-------------Loss&Opt&Acc-----------------#
with tf.name_scope("loss"):
	with tf.name_scope('cross_entropy_loss'):
		cross_entropy_loss = -tf.reduce_mean(tf.reduce_sum(label*tf.log(data_out), axis=[1]))
	
	with tf.name_scope('conditional_entropy_loss'):
		l_cond_entoropy_loss = -tf.reduce_mean(tf.reduce_sum(data_out*tf.log(data_out), axis=[1]))
		#un_cond_entorpy_loss = -tf.reduce_mean(tf.reduce_sum(un_data_out*tf.log(un_data_out), axis=[1]))

		#cond_entropy_loss = l_cond_entoropy_loss + un_cond_entorpy_loss
		cond_entropy_loss = l_cond_entoropy_loss
	
	with tf.name_scope('vat_loss'):
		#vat_loss = Get_VAT_loss(img_data, r_adv) + Get_VAT_loss(un_img_data, un_r_adv) + Affine_loss(img_data, img_data_aug)
		vat_loss = Get_VAT_loss(img_data, r_adv)

	cost = cross_entropy_loss + cond_entropy_loss + vat_loss 

with tf.name_scope("opt"):
	trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "NN")	
	for item in trainable_vars:
		print(item)	
	#optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=trainable_vars)
	adam = tf.train.AdamOptimizer(0.0002,0.5)	
	gradients_vars = adam.compute_gradients(cost, var_list=trainable_vars)	
	train_op = adam.apply_gradients(gradients_vars)


with tf.name_scope("accuracy"):
	accuracy = Accuracy(data_out, label)

#--------------Summary-----------------#
with tf.name_scope('summary'):
	with tf.name_scope('image_summary'):
		tf.summary.image('image', tf.image.convert_image_dtype(img_data, dtype=tf.uint8, saturate=True), collections=['train'])
		tf.summary.image('image_noise', tf.image.convert_image_dtype(data_r, dtype=tf.uint8, saturate=True), collections=['train'])

	with tf.name_scope("train_summary"):
		tf.summary.scalar('train_accuracy', accuracy, collections=['train'])
		tf.summary.scalar('cross_entropy_loss', cross_entropy_loss, collections=['train'])
		tf.summary.scalar('cond_entropy_loss', cond_entropy_loss, collections=['train'])
		tf.summary.scalar('vat_loss', vat_loss, collections=['train'])
		tf.summary.scalar('total_loss', cost, collections=['train'])	

	#with tf.name_scope("test_summary"):
		#acc_summary_test = tf.summary.scalar("test_accuracy", accuracy)	

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
	if a.load_model is not True:
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
			sess.run(train_op)
			if step % a.print_loss_freq == 0:
				print(step)
				sess.run(cost)
				train_acc = sess.run(accuracy)
				print("train accuracy", train_acc)
				summary_writer.add_summary(sess.run(merged), step)
				
				"""
				test_acc = 0
				test_batch_size = 32
				step_num = -(-186//test_batch_size)
				for i in range(step_num):
					print(i)
					tmp_acc = sess.run(accuracy, feed_dict={am_testing: True, drop: 0.0})
					print(tmp_acc)
					test_acc += tmp_acc
				test_acc = test_acc/step_num
				summary_writer.add_summary(tf.Summary(value=[
					tf.Summary.Value(tag="test_summary/test_accuracy", simple_value=test_acc)
				]), step)
				print("test accuracy", test_acc)
				print()
				"""
			if step % (iteration_num/5) == 0:
           		# SAVE
				saver.save(sess, a.save_dir + "/model/model.ckpt")
		
		saver.save(sess, a.save_dir + "/model/model.ckpt")
		print('saved at '+ a.save_dir)
	coord.request_stop()
	coord.join(threads)
end_time = time.time()
print( 'time : ' + str(end_time - start_time))
