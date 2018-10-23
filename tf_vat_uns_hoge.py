#vat.pyを参考に

import tensorflow as tf
import numpy as np
import datetime
import os
import time
import glob
import math
import argparse
import sys
import math

import tensorflow_hub as hub
#module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1", trainable=False)
module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1", trainable=False)

parser = argparse.ArgumentParser()
#data args
parser.add_argument("--load_model", action='store_true', help="test is do --load_model")
parser.add_argument("--load_model_path", default=None, help="path for checkpoint")
parser.add_argument("--input_file_path", help="input train data path")
parser.add_argument("--save_dir", help="path for save the model and logs") 
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epoch", type=int, help="epoch")
parser.add_argument("--print_loss_freq", type=int, default=500, help="print loss epoch frequency")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout_rate. test: 0.0, train=0.5") 
parser.add_argument("--csv_name", help="path for path_label csv file")
parser.add_argument("--sample_size", type=int, help="sample size")
parser.add_argument("--gpu_config", default=0, help="0:gpu0, 1:gpu1, -1:both")

a = parser.parse_args()
for k, v in a._get_kwargs():
    print(k, "=", v)

#------paramas------#
sample_size = a.sample_size
#sample_size = 100
n_classes = 38
iteration_num = int(a.sample_size/a.batch_size*a.epoch)
seed = 1145141919
model_size = 224
Ip = 1
xi = 1e-6
eps = 8.0
#--------function--------#
def next_batch(num, data, labels):
	idx = np.arange(0 , len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[ i] for i in idx]
	labels_shuffle = [labels[ i] for i in idx]
	return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def Accuracy(logits,label):
	correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(label,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return accuracy

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

def model(x): #ソフトマックスまで
	outputs = module(x)
   	#2set: （入力）2048（出力）1000
    #3set:（入力）1000（出力）クラス数
	with tf.variable_scope('NN',reuse=tf.AUTO_REUSE):
		logits_ = tf.layers.dense(inputs=outputs, units=1000, activation=tf.nn.leaky_relu)
		dropout_ = tf.layers.dropout(inputs=logits_, rate=dropout)
		logits = tf.layers.dense(inputs=dropout_, units=n_classes)
		out = tf.nn.softmax(logits)
		return out

#追加
def replace_none_with_zero(l):
	return [0 if i==None else i for i in l]

def Generate_perturbation(x): #画像に加えるノイズの生成
	d = tf.random_normal(shape=tf.shape(x))
	for i in range(Ip):
		d = xi * Get_normalized_vector(d)
		p = model(x)
		q = model(x + d)
		d_kl = KL_divergence(p,q)
		grad = tf.gradients(d_kl, [d], aggregation_method=2)[0] # d(d_kl)/d(d)
		if grad is None:
			grad = tf.zeros_like(img_data) #追加
		#grad = replace_none_with_zero(tf.gradients(d_kl, [d], aggregation_method=2))[0]
		print("grad", grad)
		d = tf.stop_gradient(grad)
		print(d)
	return eps * Get_normalized_vector(d)

def Get_VAT_loss(x,r): #生画像とノイズ画像の出力の差を計算
	with tf.name_scope('Get_VAT_loss'):
		p = tf.stop_gradient(model(x))
		q = model(x + r)
		loss = KL_divergence(p,q)
		return tf.identity(loss, name='vat_loss')

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
if a.gpu_config == '0':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
elif a.gpu_config == '1':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'))

start_time = time.time()
print("start time : " + str(start_time))
	
with tf.name_scope('LoadImage'):
	#csv_name = "/home/zhaoyin-t/plant_disease/traindata_int_small_random.csv" #合成あり
	#csv_name = "/home/zhaoyin-t/plant_disease/path_label_2.csv" #セグメンテーション
	csv_name = a.csv_name
	filename_queue = tf.train.string_input_producer([csv_name], shuffle=True)
	reader = tf.TextLineReader()
	_, val = reader.read(filename_queue)
	record_defaults = [["a"], ["a"], [0]]
	#record_defaults = [["a"],[0], [0], [0]]
	#path, _, label = tf.decode_csv(val, record_defaults=record_defaults)
	path, _, label = tf.decode_csv(val, record_defaults=record_defaults)
	readfile = tf.read_file(path)
	image = tf.image.decode_jpeg(readfile, channels=3)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image = tf.cast(image, dtype=tf.float32)
	
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
	image = tf.image.resize_images(image, (model_size, model_size))
	label = tf.one_hot(label, depth=n_classes)	
	label = tf.cast(label, dtype=tf.float32)
	label_batch, x_batch = tf.train.batch([label, image],batch_size=a.batch_size, allow_smaller_final_batch=False)
	iteration_num = int(sample_size/a.batch_size*a.epoch)
	#path_batch = tf.train.batch([path],batch_size=a.batch_size, allow_smaller_final_batch=False)	

	#unlabel
	unlabel_image_np = np.load("unlabel_image.npy")
	unlabel_images_tf = tf.convert_to_tensor(unlabel_image_np, tf.float32)
	unlabel_images_tf = tf.image.resize_images(unlabel_images_tf, (model_size, model_size))
	unlabel_image_queue = tf.train.input_producer(unlabel_images_tf, shuffle=False)
	unlabel_image_dequeue = unlabel_image_queue.dequeue()
	unlabel_images = tf.train.batch([unlabel_image_dequeue], batch_size=a.batch_size, allow_smaller_final_batch=True)
	

	#test
	test_image_np = np.load("test_image.npy")
	test_label_np = np.load("test_label_int_onehot.npy")
	test_images_tf = tf.convert_to_tensor(test_image_np, tf.float32)
	test_images_tf = tf.image.resize_images(test_images_tf, (model_size, model_size))
	test_labels_tf = tf.convert_to_tensor(test_label_np, tf.float32)
	test_image_queue = tf.train.input_producer(test_images_tf, shuffle=False)
	test_image_dequeue = test_image_queue.dequeue()
	test_label_queue = tf.train.input_producer(test_labels_tf, shuffle=False)
	test_label_dequeue = test_label_queue.dequeue()
	test_images, test_labels = tf.train.batch([test_image_dequeue, test_label_dequeue], batch_size=a.batch_size, allow_smaller_final_batch=True)

#--------------Model-----------------#
am_testing = tf.placeholder(dtype=bool,shape=())
img_data = tf.cond(am_testing, lambda:test_images, lambda: x_batch)
un_img_data = unlabel_images
label = tf.cond(am_testing, lambda:test_labels, lambda:label_batch)
dropout = tf.cond(am_testing, lambda:0.0, lambda:a.dropout)


with tf.name_scope("Model"):
	with tf.name_scope('model_outputs'):
		data_out = model(img_data)
		un_data_out = model(un_img_data) #追加

	with tf.name_scope('Generate_perturbation'):
        # generate perturbation
		r_adv = Generate_perturbation(img_data)
		un_r_adv = Generate_perturbation(un_img_data)

        # add perturbation onto x
		data_r = img_data + r_adv
		un_data_r = un_img_data + un_r_adv

#--------------Loss&Opt&Acc-----------------#
with tf.name_scope("loss"):
	with tf.name_scope('cross_entropy_loss'):
		cross_entropy_loss = -tf.reduce_mean(tf.reduce_sum(label*tf.log(data_out), axis=[1]))
	
	with tf.name_scope('conditional_entropy_loss'):
		l_cond_entoropy_loss = -tf.reduce_mean(tf.reduce_sum(data_out*tf.log(data_out), axis=[1]))
		un_cond_entorpy_loss = -tf.reduce_mean(tf.reduce_sum(un_data_out*tf.log(un_data_out), axis=[1]))

		cond_entropy_loss = l_cond_entoropy_loss + un_cond_entorpy_loss

	with tf.name_scope('vat_loss'):
		vat_loss = Get_VAT_loss(img_data, r_adv) #image & noise
		un_vat_loss = Get_VAT_loss(un_img_data, un_r_adv) 
		
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
	with tf.name_scope("train_accuracy"):
		train_acc = Accuracy(data_out, label)
	with tf.name_scope("test_accuracy"):
		test_acc = Accuracy(data_out, label)

#--------------Summary-----------------#
with tf.name_scope('summary'):
	with tf.name_scope('image_summary'):
		tf.summary.image('image', tf.image.convert_image_dtype(img_data, dtype=tf.uint8, saturate=True), collections=['train'])
		tf.summary.image('image_noise', tf.image.convert_image_dtype(data_r, dtype=tf.uint8, saturate=True), collections=['train'])
		tf.summary.image('unlabel_image', tf.image.convert_image_dtype(un_img_data, dtype=tf.uint8, saturate=True), collections=['train'])
		tf.summary.image('unlabel_image_noise', tf.image.convert_image_dtype(un_data_r, dtype=tf.uint8, saturate=True), collections=['train'])
	
	with tf.name_scope("train_summary"):
		tf.summary.scalar('train_accuracy', train_acc, collections=['train'])
		tf.summary.scalar('cross_entropy_loss', cross_entropy_loss, collections=['train'])
		tf.summary.scalar('cond_entropy_loss', cond_entropy_loss, collections=['train'])
		tf.summary.scalar('vat_loss', vat_loss, collections=['train'])
		tf.summary.scalar('total_loss', cost, collections=['train'])	
	
	with tf.name_scope("test_summary"):
		acc_summary_test = tf.summary.scalar("test_accuracy", test_acc)
		
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name + '/Variable_histogram', var, collections=['train'])
	
	for grad, var in gradients_vars:
		tf.summary.histogram(var.op.name + '/Gradients', grad, collections=['train'])

#---------------Session-----------------#
init = tf.global_variables_initializer()
#saver = tf.train.Saver()
tmp_config = tf.ConfigProto(
	gpu_options=tf.GPUOptions(
		visible_device_list="0",
		allow_growth=True
	)
)
saver = tf.train.Saver(trainable_vars)
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
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		for step in range(iteration_num):
		#for step in range(1):
			"""
			data = x_batch
			feed_dict_test = {am_testing:False}
			p = sess.run(model(x_batch), feed_dict_test)
			d = sess.run(tf.random_normal(shape=tf.shape(data)), feed_dict_test)
			d = sess.run(xi * Get_normalized_vector(d), feed_dict_test)
			#print(d)
			data_d = data+d
			q = sess.run(model(data_d), feed_dict_test)
			print("p", sum(p[1,]))
			print()
			print("q", sum(q[1,]))
			print()
			print(sess.run(p * tf.log(p + 1e-14), feed_dict_test))
			print()
			print(sess.run(tf.log(q + 1e-14), feed_dict_test))
			print()
			print(sess.run(KL_divergence(p,q)))
			"""
					
			sess.run(train_op, feed_dict={am_testing: False})
			if step % a.print_loss_freq == 0:
				feed_dict_train = {am_testing:False}
				feed_dict_test = {am_testing:True}
				print("step", step)	
				sess.run(cost, feed_dict=feed_dict_train)
				acc = sess.run(train_acc, feed_dict=feed_dict_train)
				print("train accuracy", acc)
				val_acc = sess.run(test_acc, feed_dict=feed_dict_test)
				print("train accuracy", val_acc)
				summary_writer.add_summary(sess.run(merged, feed_dict=feed_dict_train), step)
	
				"""
				#test
				sess.run(cost, feed_dict=feed_dict_test)
				acc_t = sess.run(test_acc, feed_dict=feed_dict_test) #名前の重複注意
				print("test accuracy", acc_t)
				print(sess.run(tf.argmax(label, 1), feed_dict=feed_dict_test))
				summary_writer.add_summary(sess.run(acc_summary_test, feed_dict=feed_dict_test), step)
				print()
				"""
				"""
				val_step = 1
				val_acc = 0
				for val_step in range( -(-186 // a.batch_size) ):
					#test_img_batch, test_label_batch = sess.run([test_images, test_labels], feed_dict=feed_dict_test)
					la = sess.run(label, feed_dict=feed_dict_test)
					print(sess.run(tf.argmax(la,1)))
					lo = sess.run(data_out, feed_dict=feed_dict_test)
					print(sess.run(tf.argmax(lo,1)))
					step_acc = acc_t = sess.run(test_acc, feed_dict=feed_dict_test) #名前の重複注意
					print("step acc", step_acc)
					val_acc += step_acc
					val_step += 1

				val_acc = val_acc/val_step
				print("test accuracy", val_acc)
				summary_writer.add_summary(tf.Summary(value=[
				    tf.Summary.Value(tag="test_summary/test_accuracy", simple_value=val_acc)
					]), step)
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
