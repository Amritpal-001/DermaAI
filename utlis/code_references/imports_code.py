import tensorflow as tf
import glob
import numpy as np
import os
import time
import PIL
from PIL import Image

from tensorflow import keras


from tensorflow.keras.utils import to_categorical
from tensorflow.python.framework.ops import Tensor
from tensorflow.keras import backend
from tensorflow.keras import applications



from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import layers
from tensorflow.keras.layers import  MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average , Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten , ZeroPadding2D , Convolution2D , MaxPooling2D ,Concatenate , Lambda


from tensorflow.keras.preprocessing.image import ImageDataGenerator , array_to_img, img_to_array, load_img

from tensorflow.keras.optimizers import Adam ,SGD
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import History, ModelCheckpoint, TensorBoard , EarlyStopping


from typing import Tuple, List

#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.applications.resnet import ResNet50 #ResNet152 ResNet101
#from tensorflow.keras.applications.resnet_v2 import ResNet152V2 #ResNet152V2 ResNet101V2
#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.densenet import DenseNet201 #DenseNet121 DenseNet169

