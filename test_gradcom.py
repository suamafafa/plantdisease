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


np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", action='store_true', help="test is do --load_model")
parser.add_argument("--load_model_path", default=None, help="path for checkpoint")
parser.add_argument("--input_file_path", help="input train data path")
parser.add_argument("--save_dir", help="path for save the model and logs")
parser.add_argument("--model", help="inception, resnet")
#parser.add_argument("--batch_size", type=int, default=32, help="batch size")
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
	
#params
seed = 1141919
n_classes = 21

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
	label_batch, x_batch = tf.train.batch([label, image],batch_size=1, allow_smaller_final_batch=False)


def model(data):
	outputs = module(data)
	with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
		#2set: （入力）2048（出力）1000
		#3set:（入力）1000（出力）クラス数
		logits_ = tf.layers.dense(inputs=outputs, units=1000, activation=tf.nn.leaky_relu, name="dense")			
		dropout_ = tf.layers.dropout(inputs=logits_, rate=0.0)
		logits = tf.layers.dense(inputs=dropout_, units=n_classes, name="output")
		return logits

#saver = tf.train.Saver()	
tmp_config = tf.ConfigProto(
	gpu_options=tf.GPUOptions(
	visible_device_list="1",
	allow_growth=True
	)
)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session(config=tmp_config) as sess:
	checkpoint = tf.train.latest_checkpoint(a.load_model_path + '/model')
	meta_path = '%s.meta' % checkpoint
	saver = tf.train.import_meta_graph(meta_path)
	saver.restore(sess, checkpoint)
	print('Loaded model from {}'.format(a.load_model_path))
	graph = tf.get_default_graph()
	for op in graph.get_operations():
		#if "logits" in op.values:
			#print(op.name) #op
			#print(op.values) #tensor
	def Grad_Cam(model, x):
		#x: image
		X = np.expand_dims(x, axis=0)
		X = X.astype('float32')
		preprocessed_input = X / 255.0

		# 予測クラスの算出
		logits = model(X)
		pred = tf.argmax(logits,1)
		class_output = logits[:, pred]

		#  勾配を取得
		conv_output = graph.get_tensor_by_name('output:0')
		grads = tf.gradients(class_output, conv_output)
		gradient_function = K.function([model.input], [conv_output, grads])
		output, grads_val = gradient_function([preprocessed_input])
		output, grads_val = output[0], grads_val[0]

		# 重みを平均化して、レイヤーのアウトプットに乗じる
		weights = np.mean(grads_val, axis=(0, 1))
		cam = np.dot(output, weights)

		# 画像化してヒートマップにして合成
		cam = cv2.resize(cam, (200, 200), cv2.INTER_LINEAR) # 画像サイズは200で処理したので
		cam = np.maximum(cam, 0) 
		cam = cam / cam.max()
		jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
		jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
		jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成	
		return jetcam

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	for i in range(1):
		#output = graph.get_tensor_by_name("train_logits/output:0")
		#output = tf.get_collection("output")
		output = tf.get_default_graph().get_tensor_by_name('train_logits/output:0')
		print(output)
		feed_dict = {data: x_batch, label: label_batch}
		print(sess.run(output, feed_dict))
		#pred = sess.run(tf.argmax(model(x_batch), 1))
		#print("output", sess.run(tf.argmax(output(x_batch), 1)))
		#print(sess.run(tf.argmax(model(x_batch), 1)))
		print(sess.run(tf.argmax(label_batch, 1)))
		#image_cam = Grad_Cam(model, image)

