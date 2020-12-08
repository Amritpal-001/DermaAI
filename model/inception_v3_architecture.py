
import os

import tensorflow
import glob
import numpy as np
from tensorflow.keras import layers , backend , applications , metrics , regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import  MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average , Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten , ZeroPadding2D , Convolution2D , MaxPooling2D \
    ,Concatenate, Lambda

from tensorflow.python.keras.utils import data_utils

from init_variables import *

'''
WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.5/'
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.5/'
                       'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def InceptionV3(input_shape, include_top=True, weights='imagenet', pooling=None):

    global backend, layers, models, backend, layers, models

    classes = 1000
    img_inputI = layers.Input(shape=input_shape)

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_inputI, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

    if include_top:
        # Classification block
        InceptionV3_second_last = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        InceptionV3_last = layers.Dense(classes, activation='softmax', name='predictions')(InceptionV3_second_last)
    else:
        if pooling == 'avg':
            InceptionV3_last = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            InceptionV3_last = layers.GlobalMaxPooling2D()(x)


    #InceptionV3_second_last = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    #InceptionV3_last = layers.Dense(classes, activation='softmax', name='predictions')(InceptionV3_second_last)

    InceptionV3_base = Model(img_inputI, InceptionV3_last, name='InceptionV3')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = data_utils.get_file( 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,   cache_subdir='models', file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
            weights_path = data_utils.get_file( 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP, cache_subdir='models', file_hash='bcbd6486424b2319ff4ef7d526e38f63')
        InceptionV3_base.load_weights(weights_path)
    elif weights is not None:
        InceptionV3_base.load_weights(weights)

    return InceptionV3_base  # , custom_InceptionV3_base_part


def custom_InceptionV3_base(num_classes, input_shape=(299, 299, 3)):
    InceptionV3_base = InceptionV3(input_shape=input_shape, include_top=True, weights='imagenet', pooling=None)
    x = InceptionV3_base.layers[-1].output
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=InceptionV3_base.input, outputs=predictions)
    # model.summary()
    return (model)'''


#model = InceptionV3(input_shape=(299, 299, 3))
def custom_original_ResNet50_base(num_classes , input_shape= None):
  resnet_base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape,
                                                       pooling='avg')
  x = resnet_base.layers[-1].output
  predictions = Dense(1, activation = "sigmoid")(x)
  model = Model(inputs = resnet_base.input,  outputs = predictions)
  model.summary()
  return(model)

def custom_original_Xception_base(num_classes , input_shape= None):
  xception_base = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=input_shape,
                                                       pooling='avg')
  x = xception_base.layers[-1].output
  predictions = Dense(1, activation = "sigmoid")(x)
  model = Model(inputs = xception_base.input,  outputs = predictions)
  model.summary()
  return(model)

def custom_original_InceptionV3_base(num_classes , input_shape= None):
  #InceptionV3_base  = InceptionV3(input_shape= input_shape , include_top=True, weights='imagenet', pooling=None)
  InceptionV3_base = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape= input_shape ,  pooling = 'avg' )
  x = InceptionV3_base.layers[-1].output
  predictions = Dense(1, activation = "sigmoid")(x)
  model = Model(inputs = InceptionV3_base.input,  outputs = predictions)
  #model.summary()
  return(model)


#model = custom_original_InceptionV3_base(num_classes = 9, input_shape= (299 , 299 , 3))


'''#model = InceptionV3(input_shape=(299, 299, 3))
def custom_original_InceptionV3_base(num_classes , input_shape= None):
  #InceptionV3_base  = InceptionV3(input_shape= input_shape , include_top=True, weights='imagenet', pooling=None)
  InceptionV3_base = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape= input_shape ,  pooling = 'avg' )
  x = InceptionV3_base.layers[-1].output
  predictions = Dense(num_classes, activation = "softmax")(x)
  model = Model(inputs = InceptionV3_base.input,  outputs = predictions)
  #model.summary()
  return(model)'''

def Freeze_model(model, freeze_till):
    freeze_till = (-1*freeze_till)
    for layer in model.layers[:freeze_till]:
        layer.trainable = False
    
    for l in model.layers:
            print(l.name, l.trainable)


def generate_defined_model(model_architecture , num_classes , input_shape= None , freeze_till = freeze_till  , draw_plot = True):
    if architecture == 'xception':
        model == custom_original_Xception_base(num_classes , input_shape= None)
    if model_architecture = 'resnet':
        model == custom_original_ResNet50_base(num_classes , input_shape= None)
        
    Freeze_model(model, freeze_till)
    
    if draw_plot == True:
        tf.keras.utils.plot_model( model, to_file= ('model' + model_architecture + '.png') , show_shapes=False, show_layer_names=True,)
    
        
def find_freeze_layer_num(model):
    a = len(model.layers) - 1
    for layer in model.layers:
        print(layer.name , '       ',  a)
           a -= 1
