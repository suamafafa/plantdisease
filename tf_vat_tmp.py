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
import random
import tensorflow_hub as hub

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
parser.add_argument("--transforming", action="store_true", help="True: Do transforming")
parser.add_argument("--gpu_config", default=1, help="0:gpu0, 1:gpu1, -1:both")

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
sample_size = 17257
#sample_size = 100
n_classes = 10
iteration_num = int(sample_size/a.batch_size*a.epoch)
seed = 1145141919
Ip = 1
xi = 1e-6
eps = 8.0
#--------function--------#
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
	with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
		logits_ = tf.layers.dense(inputs=outputs, units=1000, activation=tf.nn.leaky_relu)
		dropout_ = tf.layers.dropout(inputs=logits_, rate=dropout_rate)
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
		d = tf.stop_gradient(grad)
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
	csv_name = "/home/zhaoyin-t/plant_disease/traindata_int_small_random.csv" #合成あり
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
	image = tf.image.resize_images(image, (model_size, model_size))
	label = tf.one_hot(label, depth=n_classes)	
	label_batch, x_batch = tf.train.batch([label, image],batch_size=a.batch_size, allow_smaller_final_batch=False)

	

#--------------Model-----------------#
am_testing = tf.placeholder(dtype=bool,shape=())
img_data = tf.cond(am_testing, lambda:test_images, lambda: x_batch)
label = tf.cond(am_testing, lambda:test_labels, lambda:label_batch)
dropout_rate = tf.cond(am_testing, lambda:0.0, lambda:a.dropout)


with tf.name_scope("Model"):
	with tf.name_scope('model_outputs'):
		data_out = model(img_data)

	with tf.name_scope('Generate_perturbation'):
        # generate perturbation
		r_adv = Generate_perturbation(img_data)
        # add perturbation onto x
		data_r   = img_data + r_adv

#--------------Loss&Opt&Acc-----------------#
with tf.name_scope("loss"):
	with tf.name_scope('cross_entropy_loss'):
		label_f = tf.cast(label, dtype=tf.float32) 
		cross_entropy_loss = -tf.reduce_mean(tf.reduce_sum(label_f*tf.log(data_out), axis=[1]))

	with tf.name_scope('vat_loss'):
		vat_loss = Get_VAT_loss(img_data, r_adv) #image & noise
	
	cost = cross_entropy_loss + vat_loss
	
with tf.name_scope("opt"): 
	trainable_vars = [var for var in tf.trainable_variables()]
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
	
	with tf.name_scope("train_summary"):
		tf.summary.scalar('train_accuracy', train_acc, collections=['train'])
		tf.summary.scalar('cross_entropy_loss', cross_entropy_loss, collections=['train'])
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
		visible_device_list="1",
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

		#追加
		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		run_metadata = tf.RunMetadata()

		#merged = tf.summary.merge_all(key="train")
		merged = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(os.path.join(a.save_dir,'summary'), graph=sess.graph)
		tf.train.start_queue_runners(sess=sess)
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

				summary_writer.add_summary(sess.run(merged, feed_dict=feed_dict_train), step)
				
				#test
				print(sess.run(tf.argmax(label, 1), feed_dict=feed_dict_test))
				sess.run(cost, feed_dict=feed_dict_test)
				acc_t = sess.run(test_acc, feed_dict=feed_dict_test) #名前の重複注意
				print("test accuracy", acc_t)
				summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
				summary_writer.add_summary(sess.run(acc_summary_test, feed_dict=feed_dict_test), step)
				print()
			if step % (iteration_num/5) == 0:
              	# SAVE
				saver.save(sess, a.save_dir + "/model/model.ckpt")
		
		saver.save(sess, a.save_dir + "/model/model.ckpt")
		print('saved at '+ a.save_dir)
end_time = time.time()
print( 'time : ' + str(end_time - start_time))
