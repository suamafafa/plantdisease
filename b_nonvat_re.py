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
csv_name = 'tomato_df_train_random.csv'
csv = pd.read_csv(csv_name, header=None)
#test_csv_name = 'tomato_test_only_tomato.csv'
test_csv_name = 'tomato_df_test_random.csv'
test_csv = pd.read_csv(test_csv_name, header=None)
#path col=0 
#label col=4
sample_size = csv.shape[0]
n_class = len(np.unique(csv[4]))
seedd = 1141919

#placeholder
data = tf.placeholder(tf.float32, [None, model_size, model_size, 3])
label = tf.placeholder(tf.float32, [None, n_class])
drop = tf.placeholder(tf.float32)

#--------------Model-----------------#
#QQQ
#with tf.variable_scope('def_model', reuse=tf.AUTO_REUSE)
def model(data):
	logits_ = tf.layers.dense(inputs=module(data), units=1000)
	dropout_ = tf.layers.dropout(inputs=logits_, rate=drop)
	logits = tf.layers.dense(inputs= dropout_, units=n_class)
	out = tf.nn.softmax(logits)
	return out

with tf.name_scope('model'):
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
		y = model(data)
	
#--------------Loss&Opt-----------------#
with tf.name_scope("cost"):
	cost = -tf.reduce_mean(tf.reduce_sum(label*tf.log(y), axis=[1]))
	
with tf.name_scope("opt"): 
	#trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "trainable_section")
	trainable_vars = [var for var in tf.trainable_variables()]
	adam = tf.train.AdamOptimizer(0.0002,0.5)
	gradients_vars = adam.compute_gradients(cost, var_list=trainable_vars)	
	train_op = adam.apply_gradients(gradients_vars)

def Accuracy(y, label):
	correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(label,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return accuracy

with tf.name_scope("accuracy"):
	accuracy = Accuracy(y, label)

#--------------Summary-----------------#
with tf.name_scope('summary'):
	with tf.name_scope('image_summary'):
		tf.summary.image('image', tf.image.convert_image_dtype(data, dtype=tf.uint8, saturate=True), collections=['train'])
		tf.summary.image('image2', data, collections=['train'])
		tf.summary.image('image3', tf.image.convert_image_dtype(data*255.0, dtype=tf.uint8, saturate=True), collections=['train'])

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
		graph = tf.get_default_graph()
		placeholders = [ op for op in graph.get_operations() if op.type == "Placeholder"]
		print("placeholder", placeholders)
		step = 0
		for epo in range(a.epoch):
			csv_idx = list(range(sample_size))
			random.shuffle(csv_idx)
			for i in range(sample_size//a.batch_size):
				batch_idx = csv_idx[(i*a.batch_size):((i+1)*a.batch_size)]
				trains = np_loader(csv, batch_idx)

				sess.run(train_op, feed_dict={data:train_imgs, label:train_labels, drop:a.dropout})
			
				if step % a.print_loss_freq == 0:
					print(step)
					train_acc = sess.run(accuracy, feed_dict={data:train_imgs, label:train_labels, drop:0.0})
					print("train accuracy", train_acc)
					summary_writer.add_summary(sess.run(merged, feed_dict={data:train_imgs, label:train_labels, drop:0.0}), step)
				
					step_num = -(-test_csv.shape[0]//a.batch_size)
					tmp_acc = 0
					for i in range(step_num):
						test_batch_idx = random.sample(range(test_csv.shape[0]), a.batch_size)
						batch_test_csv = test_csv.iloc[test_batch_idx,:]
						tests = np_loader(test_csv,batch_test_csv)
						test_imgs = tests[0]
						test_labels = tests[1]

						tmp_acc += sess.run(accuracy, feed_dict={data:test_imgs, label:test_labels, drop:0.0})
					test_acc = tmp_acc/step_num
					print('test_acc', test_acc)
					summary_writer.add_summary(tf.Summary(value=[
            	    tf.Summary.Value(tag="test_summary/test_accuracy", simple_value=test_acc)]), step)
				
				if step % 500 == 0:
      				# SAVE
					saver.save(sess, a.save_dir + "/model/model.ckpt")
				step += 1
		
		saver.save(sess, a.save_dir + "/model/model.ckpt")
		print('saved at '+ a.save_dir)
		

	else: 
		print("a.load_model True")

end_time = time.time()
print( 'time : ' + str(end_time - start_time))


