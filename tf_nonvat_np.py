#without lightGBM
#回転とか切り出し
#smallを使って、100エポック

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
parser.add_argument("--dropout", type=float, default=0.2, help="dropout_rate. test: 0.0, train=0.2") 
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
n_classes = a.nclass
iteration_num = int(sample_size/a.batch_size*a.epoch)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
if a.gpu_config == '0':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
elif a.gpu_config == '1':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'))

start_time = time.time()
print("start time : " + str(start_time))

with tf.name_scope('LoadImage'):
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
	
	#train
	csv_name = "/home/zhaoyin-t/plant_disease/traindata_int_small_random_disease.csv"
	path_label = pd.read_csv(csv_name, header=None)
	
	#test
	test_image_np = np.load("test_image.npy")
	test_label_np = np.load("test_label_int_onehot.npy")
	
#---------------Model--#---------------#
#追加
def ImageDecorate(imgs): #with resize
	r = imgs
	CROP_SIZE = 256
	SCALE_SIZE = 286
	seed = 1141919
	rot90_times = tf.random_uniform([1], 0,5,dtype=tf.int32)[0]
	crop_offset = tf.cast(tf.floor(tf.random_uniform([2], 0, SCALE_SIZE - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
	r = tf.image.rot90(r, k=rot90_times)
	r = tf.image.resize_images(r, [SCALE_SIZE, SCALE_SIZE], method=tf.image.ResizeMethod.AREA)
	r = tf.image.crop_to_bounding_box(r, crop_offset[0], crop_offset[1], CROP_SIZE, SCALE_SIZE)
	param1 = 0.25
	param2 = 0.75
	r = tf.image.random_brightness(r,param1) #明るさ調整
	r = tf.image.random_contrast(r,lower=param2, upper=1/param2) #コントラスト調整
	r = tf.image.resize_images(r, [model_size, model_size])
	return r

def Resize(imgs): #resize only
	r = imgs
	r = tf.image.resize_images(r, [model_size, model_size])
	return r

#追加
data = tf.placeholder(tf.float32, [None, model_size, model_size, 3])
label = tf.placeholder(tf.float32, [None, n_classes])
dropout = tf.placeholder(tf.float32)

outputs = module(data)

with tf.variable_scope("trainable_section"):
	#2set: （入力）2048（出力）1000
	#3set:（入力）1000（出力）クラス数	
	logits_ = tf.layers.dense(inputs=outputs, units=1000, activation=tf.nn.leaky_relu)
	dropout_ = tf.layers.dropout(inputs=logits_, rate=dropout)
	logits = tf.layers.dense(inputs=dropout_, units=n_classes) 
	
#--------------Loss&Opt-----------------#
with tf.name_scope("cost"):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))

with tf.name_scope("opt"): 
	trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "trainable_section")
	#optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=trainable_vars)
	adam = tf.train.AdamOptimizer(0.0002,0.5)
	gradients_vars = adam.compute_gradients(cost, var_list=trainable_vars)	
	train_op = adam.apply_gradients(gradients_vars)
	
with tf.name_scope("correct_pred"):
	correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits),1), tf.argmax(label, 1))

with tf.name_scope("accuracy"):
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#--------------Summary-----------------#
with tf.name_scope('summary'):
	with tf.name_scope('image_summary'):
		tf.summary.image('image', tf.image.convert_image_dtype(data, dtype=tf.uint8, saturate=True), collections=['train'])
	
	with tf.name_scope("train_summary"):
		cost_summary_train = tf.summary.scalar('train_loss', cost, collections=['train'])
		acc_summary_train = tf.summary.scalar("train_accuracy", accuracy, collections=['train'])
	
	with tf.name_scope("test_summary"):
		acc_summary_test = tf.summary.scalar("test_accuracy", accuracy)
	
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
		print(tf.global_variables_initializer())
		print("Session Start")
		print("")
		merged = tf.summary.merge_all(key="train")
		summary_writer = tf.summary.FileWriter(os.path.join(a.save_dir,'summary'), graph=sess.graph)
		#coord = tf.train.Coordinator()
		#threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		step = 0
		for eph in range(a.epoch):
			for idx in range(sample_size // a.batch_size):

				batch_idx = np.random.choice(range(sample_size), a.batch_size)

				#train data load
				#path_label = #上で定義
				path_tmp = path_label.iloc[batch_idx,:]
				train_imgs = np.array([cv2.imread(im) for im in path_tmp[0]])
				train_imgs = train_imgs/255.0
				train_imgs = np.resize(train_imgs, [a.batch_size, model_size, model_size, 3])
				#diseae labelは4col
				#onehotにするために、一度col4をまとめて取り出してonehotに。
				#それをlabel_tmpとして、batch_idx分とりだす
				aa = path_label[2]
				label_tmp = np.identity(38)[aa]
				train_labels = label_tmp[batch_idx]

				#test data load
				#test_image_np = np.load("test_image.npy")
				#test_label_np = np.load("test_label_int_onehot_disease.npy")
				test_batch_idx = np.random.choice(range(168), a.batch_size)	
				test_imgs = test_image_np[test_batch_idx]
				test_imgs = np.resize(test_imgs, [a.batch_size, model_size, model_size, 3])
				test_labels = test_label_np[test_batch_idx]

				sess.run(train_op, feed_dict={data: train_imgs, label: train_labels, dropout: a.dropout})
				if step % a.print_loss_freq == 0:
					print(step)
					print(sess.run(tf.argmax(label, 1), feed_dict={data: train_imgs, label: train_labels, dropout: 0.0}))
					sess.run(cost, feed_dict={data: train_imgs, label: train_labels, dropout: 0.0})
					train_acc = sess.run(accuracy, feed_dict={data: train_imgs, label: train_labels, dropout: 0.0})
					print("train accuracy", train_acc)
					summary_writer.add_summary(sess.run(merged, feed_dict={data: train_imgs, label: train_labels, dropout: 0.0}), step)

					#test
					print(sess.run(tf.argmax(logits, 1), feed_dict={data: test_imgs, label: test_labels, dropout: 0.0}))
					print(sess.run(tf.argmax(label, 1), feed_dict={data: test_imgs, label: test_labels, dropout: 0.0}))
					test_acc = sess.run(accuracy, feed_dict={data: test_imgs, label: test_labels, dropout: 0.0})
					print("test accuracy", test_acc)
					summary_writer.add_summary(tf.Summary(value=[
						tf.Summary.Value(tag="test_summary/test_accuracy", simple_value=test_acc)
					]), step)
					print()

					step += 1

				if step % (iteration_num/5) == 0:
              		# SAVE
					saver.save(sess, a.save_dir + "/model/model.ckpt")
		
		saver.save(sess, a.save_dir + "/model/model.ckpt")
		print('saved at '+ a.save_dir)
		#coord.request_stop()
		#coord.join(threads)
end_time = time.time()
print( 'time : ' + str(end_time - start_time))
