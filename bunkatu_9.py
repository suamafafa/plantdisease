import tensorflow as tf
import numpy as np
import pandas as pd
import datetime 
import time
import os 
import glob
import math
import argparse 
import sys 
import random 
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", action='store_true', help="test is do --load_model")
parser.add_argument("--load_model_path", default=None, help="path for checkpoint")
parser.add_argument("--augm", action='store_true', help="augmentation is do")
parser.add_argument("--save_dir", help="path for save the model and logs") 
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

#config
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
if a.gpu_config == '0':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
elif a.gpu_config == '1':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'))

start_time = time.time()
print("start time : " + str(start_time))

#params 
csv_name = 'tomato_df_train_random_2_resize.csv'
#csv_name = "tomoto_and_unclass.csv"
csv = pd.read_csv(csv_name, header=None)
#test_csv_name = 'tomato_test_only_tomato.csv'
test_csv_name = 'tomato_test_only_tomato_original.csv'
test_csv = pd.read_csv(test_csv_name, header=None)
#path col=0 
#label col=1
sample_size = csv.shape[0]
n_class = len(np.unique(csv[1]))
iteration_num = int(sample_size/a.batch_size*a.epoch)
seed = 1141919

#--------------ImageLoad-----------------#
with tf.name_scope('LoadImage'):
	filename_queue = tf.train.string_input_producer([csv_name], shuffle=True)
	reader = tf.TextLineReader()
	_, val = reader.read(filename_queue)
	record_defaults = [["a"], [0], ["a"]]
	_, label, path = tf.decode_csv(val, record_defaults=record_defaults)
	readfile = tf.read_file(path)
	image = tf.image.decode_jpeg(readfile, channels=3)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image = tf.cast(image, dtype=np.float32)
	image = tf.image.resize_images(image, (256, 256))
	def noisy(img):
		img_f = tf.reshape(img, [-1])
		a_m = tf.where(img_f>1/255.0, tf.zeros_like(img_f), tf.ones(img_f.shape))
		a_m = tf.reshape(a_m, img.shape)
		noise = tf.random_uniform(a_m.shape, minval=0, maxval=1, dtype=tf.float32)
		mu = tf.multiply(a_m, noise)+img
		return mu
	with tf.name_scope('noise_images'):
		image = noisy(image)
	image = tf.image.random_brightness(image, max_delta=0.25)
	image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
	image = tf.image.adjust_hue(image, tf.random_uniform([1], minval=-0.1,maxval=0.1,dtype=tf.float32)[0])
	#image = part(image, 180, tf.random_uniform(shape=[], minval=0,maxval=1,dtype=tf.int64), tf.random_uniform(shape=[], minval=0,maxval=1,dtype=tf.int64))
	
	"""
	def contrast(img, param1, param2):
		with tf.name_scope("contrast"):
			r = img
			r = tf.image.random_brightness(r,param1) #明るさ調整
			r = tf.image.random_contrast(r,lower=param2, upper=1/param2) #コントラスト調整
			return r
	with tf.name_scope("contrast_images"):
		image = contrast(image, param1=0.1, param2=0.9)
	"""
	#label = tf.one_hot(label, depth=n_class)
	label_batch, x_batch = tf.train.batch([label, image],batch_size=a.batch_size, allow_smaller_final_batch=False)
	label_batch = tf.cast(label_batch, dtype=np.float32)

	#test
	test_filename_queue = tf.train.string_input_producer([test_csv_name], shuffle=False)
	test_reader = tf.TextLineReader()
	_, test_val = test_reader.read(test_filename_queue)
	record_defaults = [["a"], ["a"], ["a"], [0], ["a"], [0], ["a"], ["a"]]
	_, _, _, _, _, test_label, _, test_path = tf.decode_csv(test_val, record_defaults=record_defaults)
	test_readfile = tf.read_file(test_path)
	test_image = tf.image.decode_jpeg(test_readfile, channels=3)
	test_image = tf.image.convert_image_dtype(test_image, dtype=tf.float32)
	test_image = tf.cast(test_image, dtype=np.float32)
	test_image = tf.image.resize_images(test_image, (model_size, model_size),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	test_label = tf.one_hot(test_label, depth=n_class)
	test_label_batch, test_x_batch = tf.train.batch([test_label, test_image],batch_size=1, allow_smaller_final_batch=False)
	test_label_batch = tf.cast(test_label_batch, dtype=np.float32)

am_testing = tf.placeholder(dtype=bool,shape=())
label = tf.cond(am_testing, lambda:test_label_batch, lambda:label_batch)
drop = tf.placeholder(tf.float32) 

"""
def bunkatu(one_patch, size):
	one = one_patch
	for i in range(one.shape[0]):
		for j in range(one.shape[0]):
			tmp = one[i,j,]
			part = tf.reshape(tmp, [size, size, 3])
			part = tf.expand_dims(part, 0) #become 4d
			if i==0andj==0:
				parts = part
			else:
				parts = tf.concat([parts,part], 0)
	return parts
"""
#add
patch_size_t = 256*4//5
stride = 20
patch_t = tf.extract_image_patches(x_batch, ksizes=[1, patch_size_t, patch_size_t, 1], strides=[1,stride, stride, 1], rates=[1, 1, 1, 1], padding='VALID')

print("**", patch_t)

def parts(patch, size):
	for i in range(patch.shape[0]):
		one_patch = patch[i,:,:,:]
		for j in range(one_patch.shape[0]):
			for k in range(one_patch.shape[0]):
				tmp = one_patch[j,k,]
				part = tf.reshape(tmp, [size, size, 3])
				part = tf.expand_dims(part, 0) #become 4d
				if i==0 and j==0 and k==0:
					parts = part
				else:
					parts = tf.concat([parts,part], 0)
	return parts

parts_t = parts(patch_t, patch_size_t)

parts_t = tf.image.resize_images(parts_t, (model_size, model_size))


def zouhuku(item):
	return tf.fill([patch_t.shape[1]*patch_t.shape[1]], item)

#label_original = label
#label_original = tf.one_hot(label_original, depth=n_class)
label = tf.map_fn(zouhuku, label_batch)
label = tf.reshape(label,[-1])
label = tf.cast(label, dtype=np.int64)
label = tf.one_hot(label, depth=n_class)
label = tf.cast(label, dtype=np.float32)

#--------------Model-----------------#
#QQQ
with tf.variable_scope('def_model', reuse=tf.AUTO_REUSE):
	def model(data):
		logits_ = tf.layers.dense(inputs=module(data), units=1000)
		dropout_ = tf.layers.dropout(inputs=logits_, rate=drop)
		logits = tf.layers.dense(inputs= dropout_, units=n_class)
		out = tf.nn.softmax(logits)
		return out

with tf.name_scope('model'):
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
		y = model(parts_t)
		y_v = model(test_x_batch)

#--------------Loss&Opt-----------------#
with tf.name_scope("cost"):
	cost = -tf.reduce_mean(tf.reduce_sum(label*tf.log(y), axis=[1]))
	
with tf.name_scope("opt"): 
	#trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "trainable_section")
	trainable_vars = [var for var in tf.trainable_variables()]
	adam = tf.train.AdamOptimizer(0.0001,0.5)
	#adam = tf.train.AdamOptimizer(0.001,0.5)
	gradients_vars = adam.compute_gradients(cost, var_list=trainable_vars)	
	train_op = adam.apply_gradients(gradients_vars)

def Accuracy(y, label):
	correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(label,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return accuracy


def Accuracy2(y, label):
	parts_split = tf.split(y, num_or_size_splits=1) #1枚の画像にする
	for j in range(len(parts_split)):
		one_part = parts_split[j]
		regions = tf.split(one_part, num_or_size_splits=one_part.shape[0]) #同一画像内の領域群
		for i, region in enumerate(regions):
			if i==0:
				max = tf.reduce_max(region)
				pred = tf.argmax(region, 1)
			t = tf.greater(tf.reduce_max(region), max)
			if t is not None:
				pred = tf.argmax(region, 1)
				pred = tf.cast(pred, tf.float32)
		
		#答え合わせ
		label = tf.argmax(label, 1)
		answer = label[0]
		answer = tf.cast(answer, tf.float32)
		correct_pred = tf.equal(pred, answer)
		tmp_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		if j==0:
			acc = tmp_acc
		else:
			acc = tf.add(acc, tmp_acc)

	accuracy = acc/len(parts_split)
	return accuracy

def Accuracy3(y_v, label): #一個でも正解があればOK
	label = tf.argmax(label, 1) #これでshowlabelと同じ
	eq = tf.equal(tf.argmax(y_v, 1), tf.reduce_max(label))
	an = tf.reduce_any(eq)
	accuracy = tf.cast(an, tf.float32)
	return accuracy

def Pred(y):
	parts_split = tf.split(y, num_or_size_splits=1) #1枚の画像にする
	for j in range(len(parts_split)):
		one_part = parts_split[j]
		regions = tf.split(one_part, num_or_size_splits=one_part.shape[0]) #同一画像内の領域群
		for i, region in enumerate(regions):
			if i==0:
				max = tf.reduce_max(region)
				pred = tf.argmax(region, 1)
			t = tf.greater(tf.reduce_max(region), max)
			if t is not None:
				pred = tf.argmax(region, 1)
				pred = tf.cast(pred, tf.float32)
	return pred

def Test_y(y):
	parts_split = tf.split(y, num_or_size_splits=a.batch_size)
	num = np.array([])
	for i, one_image_y in enumerate(parts_split):
		y_ = tf.reshape(one_image_y, [-1])
		pred = y_[tf.argmax(y_)]
		num = np.append(num, pred)
		num_ = tf.convert_to_tensor(num, np.float32)
	return num_
def Test_label(label):
	label_split = tf.split(label, num_or_size_splits=a.batch_size)
	num = np.array([])
	for i in range(a.batch_size):
		answer = label_split[i][0]
		num = np.append(num, answer)
		num_ = tf.convert_to_tensor(num, np.float32)
		return num_

with tf.name_scope("accuracy"):
	accuracy = Accuracy(y, label)

def Showy(y):
	return tf.argmax(y,1)

def Showy_raw(y):
	return y

def Showlabel(label):
	return tf.argmax(label,1)

def Testshow(y):
	y_ = tf.reshape(y, [-1])
	return y, y_
	#return tf.argmax(y_, 1)

showy = Showy(y_v)
showlabel = Showlabel(test_label_batch)
showy_raw = Showy_raw(y_v)
#test_acc = Accuracy(y_v, label_original)
testshow = Testshow(y)[0]
testshow2 = Testshow(y)[1]
#test_y = Test_y(y)
#test_label = Test_label(label)

#--------------Summary-----------------#
with tf.name_scope('summary'):
	with tf.name_scope('image_summary'):
		tf.summary.image('parts_t', tf.image.convert_image_dtype(parts_t, dtype=tf.uint8, saturate=True), collections=['train'])
		tf.summary.image('x_batch', tf.image.convert_image_dtype(x_batch, dtype=tf.uint8, saturate=True), collections=['train'])
		tf.summary.image('test_x_batch', tf.image.convert_image_dtype(test_x_batch, dtype=tf.uint8, saturate=True), collections=['train'])
	with tf.name_scope("train_summary"):
		cost_summary_train = tf.summary.scalar('train_loss', cost, collections=['train'])
		acc_summary_train = tf.summary.scalar("train_accuracy", accuracy, collections=['train'])
	
	with tf.name_scope("test_summary"):
		acc_summary_test = tf.summary.scalar("test_accuracy", accuracy)

	for var in tf.trainable_variables():
		var_summary = tf.summary.histogram(var.op.name + '/Variable_histogram', var, collections=['train'])
	
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
		os.mkdir(os.path.join(a.save_dir,'csv'))
		sess.run(init)
		print(trainable_vars)
		print("Session Start")
		print("")
		merged = tf.summary.merge_all(key="train")
		summary_writer = tf.summary.FileWriter(os.path.join(a.save_dir,'summary'), graph=sess.graph)
		graph = tf.get_default_graph()
		placeholders = [ op for op in graph.get_operations() if op.type == "Placeholder"]
		print("placeholder", placeholders)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		for step in range(iteration_num):
			sess.run(train_op, feed_dict={am_testing: False, drop:a.dropout})		
			if step % a.print_loss_freq == 0:
				print(step)
				train_acc = sess.run(accuracy, feed_dict={am_testing: False, drop:0.0})
				print("train accuracy", train_acc)
				summary_writer.add_summary(sess.run(merged, feed_dict={am_testing:False, drop:0.0}), step)
				
				#step_num = -(-test_csv.shape[0]//a.batch_size)
				step_num = 51
				check = 0
				check_top = 0
				for i in range(step_num):
					showy_, showlabel_, softmax = sess.run([showy, showlabel, showy_raw], feed_dict={am_testing:True, drop:0.0})
					print(showy_)
					print(showlabel_)
		
					#softmax_each = np.argmax(softmax, 1) #各行の予測ラベル
					#max_softmax_value = np.array([])
					#for i in range(softmax.shape[0]):
					#	max_softmax_value=np.append(max_softmax_value, softmax[i, softmax_each[i]]) #各マックスのソフトマックスの値
					#max_value_idx = np.argmax(max_softmax_value)
					#max_value_idx = np.argpartition(max_softmax_value, -1)[-1:]
					#max_value_idx = np.argsort(max_softmax_value)[::-1]
					#preds = softmax_each[max_value_idx]
					#print(preds)
					#if np.any(preds==showlabel_[0]):
					#	check_top+=1

					if np.any(showy_==showlabel_[0]):
						check+=1
						print("check")
					#print("pred", sess.run(pred, feed_dict={am_testing:True, drop:0.0}))
					#print(sess.run(testshow, feed_dict={am_testing:True, drop:0.0}))
					#print(sess.run(testshow2, feed_dict={am_testing:True, drop:0.0}))
					print("")
					#print(sess.run(test_y, feed_dict={am_testing:True, drop:0.0}))
					#print(sess.run(test_label, feed_dict={am_testing:True, drop:0.0}))
				print("check", check/step_num)
				#print("check_top", check_top/step_num)
				summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="test_summary/test_accuracy", simple_value=check_top/step_num)]), step)

				"""
				patch_size = 112
				patch = tf.extract_image_patches(test_x_batch, ksizes=[1, patch_size, patch_size, 1], strides=[1,10, 10, 1], rates=[1, 1, 1, 1], padding='VALID')
				#patch shape = [batch_size, n,n,lots]
				#testlabel = sess.run(tf.argmax(test_label_batch,1))
				
				
				print("batch shape", patch.shape)
				num = 0
				for i in range(patch.shape[0]):
					one_patch = patch[i,:,:,:]
					for j in range(one_patch.shape[0]):
						for k in range(one_patch.shape[0]):
							tmp = one_patch[j,k,]
							part = tf.reshape(tmp, [patch_size, patch_size, 3])
							part = tf.expand_dims(part, 0) #become 4d
          				
							if j==0 & k==0:
								parts = part
							else:
								parts = tf.concat([parts,part], 0)
            	
					parts = tf.image.resize_images(parts, (model_size, model_size))
					parts = sess.run(parts)
					pred = sess.run(y, feed_dict={am_testing:True, drop:0.0, test_pl: parts})
					pred = np.argmax(pred, 1)
			
					print(pred)
					#print(testlabel[num])
					#print("")
					num += 1
				"""

			if step % 500 == 0:
      			# SAVE
				saver.save(sess, a.save_dir + "/model/model.ckpt")
		
		saver.save(sess, a.save_dir + "/model/model.ckpt")
		print('saved at '+ a.save_dir)
		
	else: 
		print("a.load_model True")

end_time = time.time()
print( 'time : ' + str(end_time - start_time))


