#test
#without lightGBM

import tensorflow as tf
import numpy as np
import datetime
import os
import time
import glob
import math
import argparse
import sys

#import tensorflow_hub as hub
#module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1", trainable=False)

parser = argparse.ArgumentParser()
#data args
parser.add_argument("--load_model", action='store_true', help="test is do --load_model")
parser.add_argument("--load_model_path", default=None, help="path for checkpoint")
parser.add_argument("--input_file_path", help="input train data path")
parser.add_argument("--save_dir", help="path for save the model and logs") 
#train args
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epoch", type=int, default=10, help="epoch")
parser.add_argument("--print_loss_freq", type=int, default=500, help="print loss epoch frequency")
parser.add_argument("--dropout", type=float, help="kepp_rate. test: 1.0, train=0.8")
#parser.add_argument("--gpu_config", default=-1, help="0:gpu, 1:gpu1, -1:both")
parser.add_argument("--gpu_config", default=0, help="0:gpu, 1:gpu1, -1:both")

a = parser.parse_args()
for k, v in a._get_kwargs():
    print(k, "=", v)

#------paramas------#
sample_size = 162915
#sample_size = 100
n_classes = 38

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
if a.gpu_config == '0':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
elif a.gpu_config == '1':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'))

start_time = time.time()
print("start time : " + str(start_time))

with tf.name_scope('LoadImage'):
	test_image_np = np.load("test_image.npy") 
	test_label_np = np.load("test_label_int.npy")
	print("test_image_np shape", test_image_np.shape)
	print("test_label_np shape", test_label_np.shape)
	test_image = tf.convert_to_tensor(test_image_int, np.float32)
	test_label = tf.convert_to_tensor(test_label_int, np.int64)


#---------------run-----------------#
#参考
#https://datascience.stackexchange.com/questions/16922/using-tensorflow-model-for-prediction















#init = tf.global_variables_initializer()
#saver = tf.train.Saver()
tmp_config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="1",
		allow_growth=True
    )
)
saver = tf.train.Saver()
with tf.Sssion(config=tmp_config) as sess:
	ckpt  = tf.train.get_checkpoint_state(a.load_model_path + "/model")
		if ckpt:
			last_model = ckpt.model_checkpoint_path
            #saver = tf.train.import_meta_graph(a.load_model_path + "/model/model.ckpt.meta")
			sver.restore(sess, last_model)
			print("load " + last_model)
			sess.run(test_image)
			sess.run(test_label)
			pred = 
			correct = 
			accuracy = 

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




saver = tf.train.Saver()
with tf.Session(config=tmp_config) as sess:
	if a.load_model is not True:
		os.mkdir(os.path.join(a.save_dir,'summary'))
		os.mkdir(os.path.join(a.save_dir,'model'))
		#sess.run(init)
		#trainable_variable_initializers = [var.initializer for var in trainable_vars]
		#sess.run(trainable_variable_initializers)
		sess.run(tf.global_variables_initializer())
		print(tf.global_variables_initializer())
		print("Session Start")
		print("")
		merged = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(os.path.join(a.save_dir,'summary'), graph=sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		for step in range(iteration_num):
		#for step in range(1):
			#print(step)
			#print("answer=", sess.run(tf.argmax(label_batch, 1)))
			#iprint(sess.run(path_batch))
			#sess.run(train_op)
			sess.run(cost)
			sess.run(optimizer)
			sess.run(train_op)
			sess.run(correct_pred)
			sess.run(accuracy)			

			summary_writer.add_summary(sess.run(merged), step)

			if step % a.print_loss_freq == 0:
				print("step", step)
				#print("cost=", sess.run(cost))
				print("logits=", sess.run(tf.argmax(logits,1)))
				print("answer=", sess.run(tf.argmax(label_batch, 1)))
				print("accuracy=%.2f" % sess.run(accuracy))
				print()
			if step % (iteration_num/5) == 0:
              	# SAVE
				saver.save(sess, a.save_dir + "/model/model.ckpt")
		
		saver.save(sess, a.save_dir + "/model/model.ckpt")
		print('saved at '+ a.save_dir)
		coord.request_stop()
		coord.join(threads)
end_time = time.time()
print( 'time : ' + str(end_time - start_time))
