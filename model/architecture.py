

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from init_variables import *
import tensorflow as tf
import numpy as np
from init_variables import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''
initializer = tf.keras.initializers.he_uniform()


if black_and_white == True:
	input_shape = (img_size,img_size,1)
else:
	input_shape = (img_size,img_size,3)


data_augmentation = tf.keras.Sequential([  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                             tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),])
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.0, offset= -1)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
#feature_batch_average = global_average_layer(feature_batch)
#print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(num_classes)
#prediction_batch = prediction_layer(feature_batch_average)
#print(prediction_batch.shape)


#Useful to deploy, as wont need additional code to preprocess the images first.
def custom_pretrained_model_with_inbuilt_preprocessing(model_architecture , input_shape ):
	if model_architecture ==  'resnet':
		base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False,weights='imagenet')
	elif model_architecture ==  'inception':
		base_model = tf.keras.applications.InceptionV3(input_shape=input_shape, include_top=False,weights='imagenet')
	elif model_architecture ==  'mobilenet':
		base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False,weights='imagenet')
	else:
		raiseError("model_arcitecute should be 'inception' or 'resnet' or 'mobilenet'")
		
	base_model.trainable = False
	
	inputs = tf.keras.Input(shape=input_shape)
	x = data_augmentation(inputs)
	x = preprocess_input(x)
	x = base_model(x, training=False)
	
	x = global_average_layer(x)
	x = tf.keras.layers.Dropout(0.2)(x)
	outputs = prediction_layer(x)
	model = tf.keras.Model(inputs, outputs)   #binary classifaciton
	return(model)'''

'''img_size = 224
input_shape = (img_size,img_size,3)
model_architecture = 'resnet'
#resnet_model = custom_pretrained_model_with_inbuilt_preprocessing(model_architecture , input_shape )
resnet_model.summary()'''


#Useful to deploy, as wont need additional code to preprocess the images first.
'''def custom_pretrained_model(model_architecture , input_shape , num_classes):
	if model_architecture ==  'resnet':
		base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=True,weights='imagenet')
	elif model_architecture ==  'inception':
		base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=True,weights='imagenet')
	elif model_architecture ==  'mobilenet':
		base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=True,weights='imagenet')
	else:
		raiseError("model_arcitecute should be 'inception' or 'resnet' or 'mobilenet'")
		
	base_model.trainable = False
	
	x = base_model(training=False)
	
	x = global_average_layer(x)
	x = tf.keras.layers.Dropout(0.2)(x)
	outputs = prediction_layer(x)
	model = tf.keras.Model(base_model.inputs, outputs)   #binary classifaciton
	return(model)


	
def get_model(model_architecuture, input_shape , pretrained = pretrained):
	if pretrained == False:
		if model_architecture ==  'resnet':
			model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, classes= num_classes , include_top=True,weights= None)
		elif model_architecture ==  'inception':
			model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,classes= num_classes , include_top=True,weights= None)
		elif model_architecture ==  'mobilenet':
			model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, classes= num_classes , include_top=True,weights= None)
		else:
			raiseError("model_arcitecute should be 'inception' or 'resnet' or 'mobilenet'")
	if pretrained == True:
		model = custom_pretrained_model(model_architecture , input_shape )
	
	return(model)'''
			



def CBR(input_shape,classes,a,b,c,d,kernel_size):
    Input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=a,kernel_size=kernel_size,padding="same", activation="relu" )(Input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
    x = tf.keras.layers.Conv2D(filters=b, kernel_size=kernel_size, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
    x = tf.keras.layers.Conv2D(filters=c, kernel_size=kernel_size, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
    x = tf.keras.layers.Conv2D(filters=d, kernel_size=kernel_size, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units= classes , activation="sigmoid")(x)

    model = Model(Input , x)
    #model.summary()
    return model

def CBRlarge(input_shape,classes):
  return CBR(input_shape,classes,64,128,256,512,(7,7))

def CBRsmall(input_shape,classes):
  return CBR(input_shape,classes,32,64,128,256,(7,7))

def CBRtiny(input_shape,classes):
  return CBR(input_shape,classes,64,128,256,512,(5,5))


def CBR_test(input_shape,classes):
  return CBR(input_shape,classes,64,128,256,512,(5,5))


print('imported architecures and generated model')
