#kerasでモデルをつくり、
#tfで使えるようにする
#vgg,alexnet,googlenetの3つくらいをやりたい

#まずはTensorflowバックエンドでKerasを使い、CNNをトレーニング
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"]
print(kerasBKED)

import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, InputLayer
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import numpy as np

#params
batch_size = 32
num_classes = 38
epoch = 500
saveDir = ""
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)


