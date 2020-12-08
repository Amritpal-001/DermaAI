# from tensorflow.keras.applications.resnet import ResNet50 #ResNet152 ResNet101
# tf.config.list_physical_devices('GPU')
import numpy as np
import tensorflow as tf


def set_random_seed(seed_value):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print('Random_seed set for', seed_value)


import os

ROOT_DIR = os.path.dirname(os.path.abspath("train_model.py"))
print(ROOT_DIR)

############################################################################
###################  Seed setting  ######################################
seed_value = 10
numpy_seed_value = 2
tensorflow_seed_value = 2

#########################################################################
###################   Model details #####################################
num_classes = 1
optimizer = 'adam'
metrics = ['accuracy', 'categorical_crossentropy']
model_architecture_list = ['xception' , 'resnet', 'inception', 'vgg', 'cbr']
model_architecture = model_architecture_list[0]
pretrained = True

if model_architecture = 'xception'
    freeze_till = 48   #XceptionNet
               # 94   #INceptionV3

#########################################################################
#################### Learning rate  #####################################

# learning_rate_list = [0.001 , 0.0005 , 0.0001 ,0.00005 , 0.00001]
learning_rate = 0.0001
Initial_DL_rate = 0.001
Final_DL_rate = 0.0001

#########################################################################
#################### Training details  ##################################
img_size  = 224


if colab == True:
    batch_size = 32
else:
    batch_size = 4
    
    

image_type = 'L'  # RGB    or   L
black_and_white = False


# data_directory = ROOT_DIR + '/data/Test'
data_directory = ROOT_DIR + '/data/Test'
epoch = 2
patience = 10

########################################################################
###################Image Preprocessing##################################
# APPLIES TO BOTH TRAIN AND VAL DATA(NOT IDEAL THING TO DO)
validation_split = 0.3
rescale = 1.0 / 255.0  # 0 / 1           # Normalize Images
rotation_range = 10
width_shift_range = 0.1
zoom_range = 0.1
horizontal_flip = True
fill_mode = 'nearest'
# Feature-wise standardization
featurewise_center = True  # Centering -refers to as featurewise_center (calculate  mean pixel value across the entire training dataset, then subtract it from each image.
featurewise_std_normalization = True  # Standardisation of Image data
# sample-wise standardization
# datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
samplewise_center = True  # Sample-wise centering  - mean pixel value for each image

######################################################################
################### Test image ######################################
test_image = 'img.jpg'
image_directory = 'img.jpg'

############################################################################
###################  Model saving details ##################################
save_location = ROOT_DIR + '/saved_results'
Saving_threshold = 70

############################################################################
###################  Filter visualisation ##################################
cmap_type_list = ['viridis', 'gray', 'RdBu', 'Blues']
cmap_type = cmap_type_list[0]


############################################################################
################### Saved_weights_info ######################################
saved_resnet_weights_directory = '/home/amritpal/PycharmProjects/100-days-of-code/100_days_of_code/Skin_lesions_Classification-master/saved_model/model_name_RGB_224__0.55__1.61__12-3_2:59'
saved_inception_weights_directory = '/home/amritpal/PycharmProjects/100-days-of-code/100_days_of_code/Skin_lesions_Classification-master/saved_model/model_name_RGB_224__0.55__1.61__12-3_2:59'
saved_vgg_weights_directory = '/home/amritpal/PycharmProjects/100-days-of-code/100_days_of_code/Skin_lesions_Classification-master/saved_model/model_name_RGB_224__0.55__1.61__12-3_2:59'
saved_cbr_weights_directory = '/home/amritpal/PycharmProjects/100-days-of-code/100_days_of_code/Skin_lesions_Classification-master/saved_model/model_name_RGB_224__0.55__1.61__12-3_2:59'

print('imported initilization_variables')

























