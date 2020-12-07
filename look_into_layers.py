	
# visualize feature maps output from each block in the vgg model
import time
import os
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
import tensorflow

from datetime import date , datetime

from init_variables import *


physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

#model_architecture = 'vgg'
#image_directory = test_image

def load_model_for_filter_visualsation(model_architecture):
	if model_architecture == 'vgg':
		model = VGG16()
		#model = tensorflow.keras.models.load_model(saved_vgg_weights_directory)
		ixs = [2, 5, 9, 13, 17]
		ixs_names_list = []
		# has 64 maps

	if model_architecture == 'resnet':
		model = ResNet50()
		#model = tensorflow.keras.models.load_model(saved_resnet_weights_directory)
		#ixs = [15, 26, 36, 47,58,68, 89, 100, 110,120,130, 140, 151 , 162,172]
		ixs = [14, 25, 35, 46 ,57,67, 88, 99, 109,119,129, 139, 150 , 161,171]
		ixs_names_list = ['conv2_block1_3_conv' ,'conv2_block2_3_conv' ,'conv2_block3_3_conv' ,
		                  'conv3_block1_3_conv' ,'conv3_block2_3_conv' ,'conv3_block3_3_conv' , 'conv3_block4_3_conv' ,
		                  'conv4_block1_3_conv' ,'conv4_block2_3_conv' ,'conv4_block3_3_conv' , 'conv4_block4_3_conv' ,'conv4_block5_3_conv' ,'conv4_block6_3_conv' , 
		                  'conv5_block1_3_conv' ,'conv5_block2_3_conv' ,'conv5_block3_3_conv' ]
	if model_architecture == 'inception':
		model = tensorflow.keras.models.load_model(saved_inception_weights_directory)
		ixs = [32, 55, 77, 123, 155 , 187 , 219 , 263 , 294]
		ixs_names_list = []
	
	if model_architecture == 'cbr':
		model = tensorflow.keras.models.load_model(saved_cbr_weights_directory)
		ixs = [2, 5, 9, 13, 17]
		ixs_names_list = []	
	return(model , ixs , ixs_names_list)
	
model , ixs , ixs_names_list = load_model_for_filter_visualsation(model_architecture)



def look_into_layers(model , image_directory , ixs):
	
	
	#creating folder to store images
	today = date.today()
	Day_part = today.strftime("%b-%d")
	now = datetime.now()
	Time_part = now.strftime("%H:%M:%S")

	
	folder_name = model_architecture +  str(Day_part) + str(Time_part) 
	model_save_path = 'model_layers/' + folder_name 
	os.mkdir(model_save_path)

	# redefine model to output right after the first hidden layer
	outputs = [model.layers[i].output for i in ixs]
	model = Model(inputs=model.inputs, outputs=outputs)

	# load the image with the required shape
	img = load_img(image_directory, target_size=(224, 224))
	img = img_to_array(img)
	# expand dimensions so that it represents a single 'sample'
	img = expand_dims(img, axis=0)
	img = preprocess_input(img)

	# get feature map for first hidden layer
	feature_maps = model.predict(img)
	
	image_save_path_list = []

	# plot the output from each block
	square = 8
	fmap_num = 0
	for fmap in feature_maps:
		# plot all 64 maps in an 8x8 squares
		ix = 1
		for _ in range(square):
			for _ in range(square):
				# specify subplot and turn of axis
				ax = pyplot.subplot(square, square, ix)
				ax.set_xticks([])
				ax.set_yticks([])
				# plot filter channel in grayscale
				pyplot.imshow(fmap[0, :, :, ix-1], cmap=cmap_type)    #'gray'
				ix += 1

		# show the figure
		#pyplot.show()
		#image_name = model_save_path + '/' + str(ix) + str(time.time()) + '.png'
		image_save_path = model_save_path + '/' +  ixs_names_list[fmap_num] + str(ixs[fmap_num]) + '.png'

		pyplot.savefig(image_save_path)
		image_save_path_list.append(image_save_path)
		fmap_num += 1

	
	return(image_save_path_list)

image_save_path_list = look_into_layers(model , image_directory, ixs )
print(image_save_path_list)























