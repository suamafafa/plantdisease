import tensorflow as tf
import numpy as np
import datetime
import os
import time
import glob
import math
import argparse
import sys
sys.path.append("/home/zhaoyin-t/function")
from Plot_data_on_image import *

parser = argparse.ArgumentParser()
#data args
parser.add_argument("--load_model", action='store_true', help="train is ignote, test is do --load_model")
parser.add_argument("--load_model_path", default=None, help="path for checkpoint")
parser.add_argument("--input_file_path", help="input train data path")
parser.add_argument("--save_dir", help="path for save the model and logs")
#train args
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epoch", type=int, default=500, help="epoch")
parser.add_argument("--print_loss_freq", type=int, default=500, help="print loss epoch frequency")
parser.add_argument("--gpu_config", default=-1, help="0:gpu, 1:gpu1, -1:both")

a = parser.parse_args()
for k, v in a._get_kwargs():
	print(k, "=", v)

n_classes = 2
sample_size = 100
image_size = 256
n_input = image_size*image_size
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, image_size, image_size, None])
y = tf.placeholder(tf.int64, [None])
dropout = 0.8

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
if a.gpu_config == '0':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
elif a.gpu_config == '1':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'))

start_time = time.time()
print("start time : " + str(start_time))


with tf.name_scope('LoadImage'):
	csv_name = "/home/zhaoyin-t/plant_disease/test2.csv"
	#if a.load_model is not True:
	filename_queue = tf.train.string_input_producer([csv_name], shuffle=True)
	#else:
	#	filename_queue = tf.train.string_input_producer([csv_name], num_epochs=1, shuffle=False)
	reader = tf.TextLineReader()
	key, val = reader.read(filename_queue)
	
	val_defaults = [["aa"], [1], [1], [1]]
	path, label, label_disease, label_plant = tf.decode_csv(val, record_defaults=val_defaults)
	readfile = tf.read_file(path)
	image = tf.image.decode_jpeg(readfile, channels=3)
	image.set_shape([256,256,3])
	#print(image.shape)
	image = tf.image.resize_images(image, [image_size, image_size])
	image = tf.cast(image, dtype=np.float32)
	image = image/255.0
	#when the target is "label"
	label_batch, x_batch = tf.train.batch([label, image],batch_size=a.batch_size) 
	iteration_num = int(sample_size/a.batch_size*a.epoch)

#with tf.name_scope("Make_input"):
#	x_batch = tf.image.resize_images(x_batch, [n_input])

#---------------Model-----------------#
# Create AlexNet model
def conv2d(name, l_input, w, b):
	#change strides due to 256*256 input
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 2, 2, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _weights, _biases, _dropout):
	# Reshape input picture
	_X = tf.reshape(_X, shape=[a.batch_size, image_size, image_size, 3])
	print(_X.shape)
	# Convolution Layer
	conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
	# Max Pooling (down-sampling)
	pool1 = max_pool('pool1', conv1, k=2)
	# Apply Normalization
	norm1 = norm('norm1', pool1, lsize=4)
	# Apply Dropout
	norm1 = tf.nn.dropout(norm1, _dropout)
	#print("norm1 shape", norm1.shape)

	# Convolution Layer
	conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
	# Max Pooling (down-sampling)
	pool2 = max_pool('pool2', conv2, k=2)
	# Apply Normalization
	norm2 = norm('norm2', pool2, lsize=4)
	# Apply Dropout
	norm2 = tf.nn.dropout(norm2, _dropout)
	#print("norm2 shape", norm2.shape)
	
	# Convolution Layer
	conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
	# Max Pooling (down-sampling)
	pool3 = max_pool('pool3', conv3, k=2)
	# Apply Normalization
	norm3 = norm('norm3', pool3, lsize=4)
	# Apply Dropout
	norm3 = tf.nn.dropout(norm3, _dropout)
	#print("norm3 shape", norm3.shape)	
	
	# Fully connected layer
	dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
	dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation
	#print("dense1 shape", dense1.shape)	
	dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
	
	# Output, class prediction
	out = tf.matmul(dense2, _weights['out']) + _biases['out']
	#print(out.shape)
	return out

# Store layers weight & bias
weights = {
   	'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
  	'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
   	'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#construct model
pred = alex_net(x, weights, biases, keep_prob) 

#---------------Loss&Opt etc------------------#
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y)
optimizer = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#----------------Summary--------------#
with tf.name_scope('Input_image_summary'):
	#tf.summary.image('Input_image', tf.image.convert_image_dtype(x_batch, dtype=tf.uint8, saturate=True))	
   
	with tf.name_scope("Loss_summary"):
		tf.summary.scalar("cost", cost)
		tf.summary.scalar("accuracy", accuracy)
 
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name + '/Variable_histogram', var)

	##for grad, var in ae_gradients_vars + dis_y_gradients_vars + dis_z_gradients_vars + cluster_head_gradients_vars:
		##tf.summary.histogram(var.op.name + '/Gradients', grad)

parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

#---------------Session-----------------#
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
	if a.load_model is not True:
		sess.run(init)
		print('total parameters : ',sess.run(parameter_count))

        # mkdir if not exist dir
		if not os.path.exists(a.save_dir):
			##os.mkdir(a.save_dir)
			os.mkdir(os.path.join(a.save_dir,'summary'))
			##os.mkdir(os.path.join(a.save_dir,'variables'))
			os.mkdir(os.path.join(a.save_dir,'model'))
	
        # remove old summary if already exist
		##if tf.gfile.Exists(os.path.join(a.save_dir,'summary')):   # NOT CHANGE
			##tf.gfile.DeleteRecursively(os.path.join(a.save_dir,'summary'))

        # merging summary & set summary writer
		merged = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(os.path.join(a.save_dir,'summary'), graph=sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		#print(label.eval())
		#print(y)
		#sess.run(pred, feed_dict={x: x_batch.eval(), keep_prob: dropout})
		#sess.run(correct_pred, feed_dict={x: x_batch.eval(), y: label_batch.eval(), keep_prob: 1.})
		#print(sess.run(accuracy, feed_dict={x: x_batch.eval(), y: label_batch.eval(), keep_prob: 1.}))

        # train
		for step in range(iteration_num):
			run_x = sess.run(x_batch)
			run_y = sess.run(label_batch)
			feed_dict = {x: run_x, y: run_y, keep_prob: dropout}
			sess.run(optimizer, feed_dict=feed_dict)
			if step % a.print_loss_freq == 0:
				feed_dict2 = {x: run_x, y: run_y, keep_prob: 1.}
				correct = sess.run(correct_pred, feed_dict=feed_dict2)
				acc = sess.run(accuracy, feed_dict=feed_dict2)
				loss = sess.run(cost, feed_dict=feed_dict2)			
				#print("Iter " + str(step % a.print_loss_freq) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
				print(correct)
				print("accuracy=%.2f" % acc)
				print()
				#summary_writer.add_summary(sess.run(merged), step)
			if step % (iteration_num/5) == 0:
                # SAVE
				saver.save(sess, a.save_dir + "/model/model.ckpt")
		saver.save(sess, a.save_dir + "/model/model.ckpt")
		print('saved at '+ a.save_dir)
		coord.request_stop()
		coord.join(threads)

	else:
		print(sess.run(val.eval()))
		#checkpoint = tf.train.latest_checkpoint(a.load_model_path + '/model')
		ckpt  = tf.train.get_checkpoint_state(a.load_model_path + "/model")
		if ckpt:
			last_model = ckpt.model_checkpoint_path
			#saver = tf.train.import_meta_graph(a.load_model_path + "/model/model.ckpt.meta") 
			saver.restore(sess, last_model)
			print("load " + last_model)
			print("wd1 : %s" % weights["wd1"].eval())
				#run_x = sess.run(x_batch)
				#run_y = sess.run(label_batch)
				#feed_dict = {x: x_batch.eval(),  y: label_batch.eval(), keep_prob: 1.0}
				#sess.run(pred, feed_dict=feed_dict)
			#correct = sess.run(correct_pred, feed_dict={x:x_batch.eval(),y:label_batch.eval(), keep_prob: 1.})
			#sess.run(accuracy, feed_dict={x:x_batch.eval(),y:label_batch.eval(), keep_prob: 1.}) #仮
			#print("結果：{:.2f}%".format(acc * 100))
			print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: x_batch.eval(), y: label_batch.eval(), keep_prob: 1.}))
		else: 
			print("no model")

end_time = time.time()
print( 'time : ' + str(end_time - start_time))
