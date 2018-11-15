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
parser.add_argument("--print_loss_freq", type=int, default=50, help="print loss epoch frequency")
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
test_csv_name = "tomato_test_only_tomato.csv"
test_csv = pd.read_csv(test_csv_name, header=None)
#path col=0 
#label col=4
sample_size = csv.shape[0]
n_class = len(np.unique(csv[4]))
seedd = 1141919

#placeholder
data = tf.placeholder(tf.float32, [None, 256, 256, 3])
data_aug = tf.placeholder(tf.float32, [None, 256, 256, 3])
label = tf.placeholder(tf.float32, [None, n_class])
drop = tf.placeholder(tf.float32)

data_resize = tf.image.resize_images(data, [model_size, model_size]) 
data_aug_resize = tf.image.resize_images(data_aug, [model_size, model_size])
#function
def ransu(k):
	return np.random.randint(0, k)

def ransu2(k):
	return np.random.randint(-k, k)

def afine(img, k=50):
    #img = cv2.imread(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	rows,cols,ch = img.shape
	pts1 = np.float32([[0,0],[0,256],[256,0],[256,256]])

	lt = [ransu2(k), ransu2(k)]
	rt = [256-ransu2(k), ransu2(k)]
	lb = [ransu2(k), 256-ransu2(k)]
	rb = [256-ransu2(k), 256-ransu2(k)]
	pts2 = np.float32([lt,lb,rt,rb])

	M = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(img,M,(256,256))
	return dst

def moment(matrix):
	mask = np.zeros((256, 256))
	for x in range(256):
		for y in range(256):
			if sum(matrix[x][y]) < 30:
				mask[x][y] = np.array(0) 
			else:
				mask[x][y] = np.array(255) 
	mu = cv2.moments(mask, False)
	x,y= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
	return x,y

def rotation(img, center, angle, scale):
	center = tuple(np.array(center)+(ransu(30),ransu(30)))
	rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
	img_dst = cv2.warpAffine(img, rotation_matrix, (256,256))
	return img_dst

def makemask(matrix):
	mask = np.zeros((256, 256, 3))
	for x in range(256):
		for y in range(256):
			if sum(matrix[x][y]) < 30:
				mask[x][y] = np.array([0, 0, 0]) # Black pixel if no object
			else:
				mask[x][y] = np.array([255, 255, 255]) 
	return mask

def overlay(foreground, background):
    # Convert uint8 to float
	foreground = foreground.astype(float)
	background = background.astype(float)
    
	mask = makemask(foreground)
 
    # Normalize the alpha mask to keep intensity between 0 and 1
	mask = mask.astype(float)/255
 
    # Multiply the foreground with the alpha matte
	foreground = cv2.multiply(mask, foreground)
 
    # Multiply the background with ( 1 - alpha )
	background = cv2.multiply((1-mask), background)

    # Add the masked foreground and background.
	outImage = cv2.add(foreground, background)
	outImage = outImage.astype('uint8')
    
	return outImage

imagenet = glob.glob("/home/zhaoyin-t/imagenet/*")
print("imagenet", imagenet[0], len(imagenet))

def gousei(img):	
	#合成
	img = afine(img)
	#center = moment(img)
	#vv = random.uniform(0.8,1.0)
	#rot_img = rotation(img, moment(img), ransu(360.0), vv)
	
	back_img = imagenet[ransu(len(imagenet))]
	back = cv2.imread(back_img)
	back = cv2.cvtColor(back,cv2.COLOR_BGR2RGB)
	outImage = overlay(img, back)
	#outImage = cv2.resize(outImage, (model_size, model_size))
	outImage = (outImage - np.mean(outImage))/np.std(outImage)*16+64
	return outImage

def Get_normalized_vector(d):
	with tf.name_scope('get_normalized_vec'):
		d /= (1e-12 + tf.reduce_sum(tf.abs(d), axis=[1,2,3], keep_dims=True))
		d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.),axis=[1,2,3], keep_dims=True))
		return d

def KL_divergence(p, q):
    # KL = 竏ｫp(x)log(p(x)/q(x))dx
    #    = 竏ｫp(x)(log(p(x)) - log(q(x)))dx
	kld = tf.reduce_mean(tf.reduce_sum(p * (tf.log(p + 1e-14) - tf.log(q + 1e-14)), axis=[1]))
	return kld

def Get_Affine_loss(x,aug):
	with tf.name_scope('Get_Affine_loss'):
		p = tf.stop_gradient(model(x))
		q = model(aug)
		loss = KL_divergence(p,q)
		return tf.identity(loss, name='affine_loss')


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
		y = model(data_resize)

#--------------Loss&Opt-----------------#
with tf.name_scope("loss"):
	with tf.name_scope("cross_entropy_loss"):
		cross_en_loss = -tf.reduce_mean(tf.reduce_sum(label*tf.log(y), axis=[1]))
	#with tf.name_scope('conditional_entropy_loss'):
		#cond_en_loss = -tf.reduce_mean(tf.reduce_sum(y*tf.log(y), axis=[1]))
	with tf.name_scope('affine_loss'):
		affine_loss = Get_Affine_loss(data_resize, data_aug_resize)
	
	cost = cross_en_loss + affine_loss

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

	with tf.name_scope("train_summary"):
		cost_summary_train = tf.summary.scalar('train_loss', cost, collections=['train'])
		acc_summary_train = tf.summary.scalar("train_accuracy", accuracy, collections=['train'])
	
	with tf.name_scope("test_summary"):
		acc_summary_test = tf.summary.scalar("test_accuracy", accuracy)

	for var in tf.trainable_variables():
		var_summary = tf.summary.histogram(var.op.name + '/Variable_histogram', var, collections=['train'])
	
	#for grad, var in gradients_vars:
		#grad_summary = tf.summary.histogram(var.op.name + '/Gradients', grad, collections=['train'])


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
				batch_csv = csv.iloc[batch_idx,:]
				train_imgs = np.array([cv2.imread(im) for im in batch_csv[0]])
				train_imgs_aug = np.array([gousei(im) for im in train_imgs])
				train_imgs = train_imgs.astype(np.float64)/255.0
				aa = csv[4]
				aa_label = np.identity(n_class)[aa]
				train_labels = aa_label[batch_idx]
			
				sess.run(train_op, feed_dict={data:train_imgs, data_aug: train_imgs_aug, label:train_labels, drop:a.dropout})
			
				if step % a.print_loss_freq == 0:
					print(step)
					train_acc = sess.run(accuracy, feed_dict={data:train_imgs, data_aug: train_imgs_aug,label:train_labels, drop:0.0})
					print("train accuracy", train_acc)
					summary_writer.add_summary(sess.run(merged, feed_dict={data:train_imgs, data_aug: train_imgs_aug,label:train_labels, drop:0.0}), step)
					
					#test
					test_imgs = np.array([cv2.imread(im) for im in test_csv[1]])
					test_imgs = test_imgs/255.0
					bb = test_csv[5]
					test_labels = np.identity(n_class)[bb]

					test_acc = sess.run(accuracy, feed_dict={data:test_imgs, data_aug: train_imgs_aug, label:test_labels, drop:0.0})
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


