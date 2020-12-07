

import tensorflow as tf
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)


from init_variables import *



if black_and_white == True:
	input_shape = img_size,img_size,1)
else:
	input_shape = img_size,img_size,3)


data_augmentation = tf.keras.Sequential([  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


def custom_pretrained_model(model_architecture , input_shape ):
	if model_architecture ==  'resnet':
		base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=False,weights='imagenet')
	elif model_architecture ==  'inception':
		base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False,weights='imagenet')
	elif model_architecture ==  'mobilenet':
		base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False,weights='imagenet')
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
	return(model)

	
pretrained = True

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
	
	return(model)

			

