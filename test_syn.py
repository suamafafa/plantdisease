import tensorflow as tf
import numpy as np
import os 
import time
import pandas as pd
import glob
import math 
import argparse
import sys
import cv2
import urllib.request
from PIL import Image
import io
import requests
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

sample_size = 100

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
if a.gpu_config == '0':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
elif a.gpu_config == '1':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'))

start_time = time.time()
print("start time : " + str(start_time))

def sys(img1, img2, name):
    #img2=background
    #ぼかす
    img2.flags.writeable = True		

    img1 = cv2.blur(img1,(10,10))

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[0:rows, 0:cols ] = dst

    pilImg = Image.fromarray(np.uint8(img1))
    pilImg.save("/home/zhaoyin-t/plant_disease/traindata/"+name+".jpg")

#csv_name1 = "/home/zhaoyin-t/plant_disease/test2.csv" 
csv_name1 = "/home/zhaoyin-t/plant_disease/path_label.csv"

csv1 = pd.read_csv(csv_name1)
#csv2 = pd.read_csv(csv_name2)

#個別対応
#1371, 6028, 21264, 34890, 35274
#i = 0
i = 6028
j = 0
for url in csv1.iloc[:,1][i:]:
	#img_read = urllib.request.urlopen(url).read()
	#img_bin = io.BytesIO(img_read)
	if i==6028 or i==21264 or i==34890 or i==35274:
		i += 1
		continue
	print(i)
	img = Image.open(url)
	img1 = np.asarray(img)  
	for filename in glob.glob('/home/zhaoyin-t/imagenet/*.jpg'):
		img = Image.open(filename)
		img2 = np.asarray(img)
		name = str(i)+"_"+str(j)
		sys(img2, img1, name)
		if j==29:
			j=0
			break
		j += 1
	i += 1

"""
img1 = tf.placeholder(tf.float32, [None, 256, 256, 3])
img2 = tf.placeholder(tf.float32, [None, 256, 256, 3])
#img1 = tf.placeholder(dtype=tf.string)
#img2 = tf.placeholder(dtype=tf.string) 
name = tf.placeholder(dtype=tf.string)


with tf.name_scope('LoadImage'):
	csv_name = "/home/zhaoyin-t/plant_disease/test2.csv"
    #if a.load_model is not True:
	filename_queue = tf.train.string_input_producer([csv_name], shuffle=True)
    #else:
    #   filename_queue = tf.train.string_input_producer([csv_name], num_epochs=1, shuffle=False)
	reader = tf.TextLineReader()
	key, val = reader.read(filename_queue)
	val_defaults = [["aa"], [1], [1], [1]]
	path, label, label_disease, label_plant = tf.decode_csv(val, record_defaults=val_defaults)
	readfile = tf.read_file(path)
	image = tf.image.decode_jpeg(readfile, channels=3)
	image.set_shape([256,256,3])
    #image = tf.image.resize_images(image, [image_size, image_size])
	#image = tf.cast(image, dtype=np.float32)
	#image = image/255.0
    #when the target is "label"
	#iteration_num = int(sample_size/a.batch_size*a.epoch)
	
	#合成用画像
	input_paths = glob.glob(os.path.join("/home/zhaoyin-t/imagenet/", "*.jpg"))
	path_queue = tf.train.string_input_producer(input_paths)
	image_reader = tf.WholeFileReader()
	key_sys, image_file = image_reader.read(path_queue)
	# adjust constant value corresponding to your paths if you face issues. It should work for above format.
	image_sys = tf.image.decode_png(image_file)	
	image_sys.set_shape([256,256,3])
	#image_sys = tf.cast(image_sys, dtype=np.float32)
	#image_sys = image_sys/255.0
	print("image_sys", image_sys)

def pre_sys(url):
	img_read = urllib.request.urlopen(url).read()
	img_bin = io.BytesIO(img_read)
	img = Image.open(img_bin)
	image = np.asarray(img)	
	return image

def sys(img2, img1, name):
	#img2=background
	#ぼかす
	#img1 = Image.open(img1)
	#img2 = np.asarray(img2)
	img2 = cv2.blur(img2,(10,10))
	
	# I want to put logo on top-left corner, So I create a ROI
	rows,cols,channels = img2.shape
	roi = img1[0:rows, 0:cols ]

	# Now create a mask of logo and create its inverse mask also
	img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)

	# Now black-out the area of logo in ROI
	img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

	# Take only region of logo from logo image.
	img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

	# Put logo in ROI and modify the main image
	dst = cv2.add(img1_bg,img2_fg)
	img1[0:rows, 0:cols ] = dst

	pilImg = Image.fromarray(np.uint8(img1))
	pilImg.save("/home/zhaoyin-t/plant_disease/traindata/"+name+".jpg")

#sys_run = sys(img1, img2, name)

#---------------Session-----------------#
init = tf.global_variables_initializer()
#saver = tf.train.Saver()
with tf.Session(config=config) as sess:
	sess.run(init)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	#tf.train.start_queue_runners(sess)

	for step in range(sample_size):
		#print(sess.run(key))
		for bc_image in range(50): #50枚ある
			name = str(step)+"_"+str(bc_image)
			#sess.run(sys, feed_dict={'DecodeJpeg:0': image.eval(), 'DecodeJpeg:1': image_sys.eval(), name: name})
			run_image = sess.run(image)
			run_image_sys = sess.run(image_sys)
			#print(run_image)
			sess.run(sys(run_image, run_image_sys, name))
		break
	coord.request_stop()
	coord.join(threads)
"""
end_time = time.time()
print( 'time : ' + str(end_time - start_time))
