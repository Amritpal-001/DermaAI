import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2
#from keras.applications.vgg16 import VGG16
from PIL import ImageTk, Image
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
import time
from init_variables import *

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

from model.inception_v3_architecture import *

#from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

def Sort_Tuple(tup):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    tup.sort(key=lambda x: x[1] , reverse= True)
    return tup


def decode_predictions_for_custom_model(preds, top=9):
  global CLASS_INDEX
  CLASS_INDEX = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']

  results = []
  print(preds)   #[[0.1120417  0.11041853 0.1125875  0.11015146 0.11088967 0.1108565  0.11060815 0.11107314 0.11137333]]
  for pred in preds:
     #print(pred)  # [0.1120417  0.11041853 0.1125875  0.11015146 0.11088967 0.1108565 0.11060815 0.11107314 0.11137333]
     top_indices = pred.argsort()[-top:][::-1]
     print(top_indices)
     for i in top_indices:
        result = [CLASS_INDEX[i] , pred[i]]
        #result = [ (pred[i])]
        results.append(result)
        #print(results)
     results = Sort_Tuple(results)
  return results

from tensorflow.python.keras.utils import data_utils
import json

CLASS_INDEX = None
fpath = 'my_imagenet_class_index.json'

def decode_predictions(preds, num_classes ,fpath,  top=5):
  global CLASS_INDEX
  if len(preds.shape) != 2 or preds.shape[1] != num_classes:
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples,',num_classes ,' )). '
                     'Found array with shape: ' + str(preds.shape))
  if CLASS_INDEX is None:
    with open(fpath) as f:
      CLASS_INDEX = json.load(f)
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results

############## in google colab only
# from google.colab.patches import cv2_imshow
# model = custom_pretrained_model_with_inbuilt_preprocessing(model_architecture , input_shape )

def gradCAM(image_directory, layer_index, img_size, model_architecture, intensity=intensity,
            gradcam_save_res=gradcam_save_res):
    #if model_architecture == 'inception':
    img_size = 299
    layer_index_list = [-15, -20, -95, -127, -159]
    layer_index = layer_index_list[1]

    #model = InceptionV3(input_shape=(299, 299, 3))
    num_classes = 9
    model = custom_original_InceptionV3_base(num_classes=num_classes, input_shape=(299, 299, 3))

    img = image.load_img(image_directory, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)  # preprocess_input
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.0, offset=-1)(x)  # rescale

    # x = preprocess_input(x)
    preds = model.predict(x)

    #label = decode_predictions(preds,  num_classes , fpath)[0][0][1] # prints the class of image
    results = decode_predictions(preds,  num_classes , fpath)
    #print(label)
    #print(len(label))
    #print(label)

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(index=layer_index)
        print(last_conv_layer.name)
        label = model_architecture + ' ' + str(last_conv_layer.name) + ' ' + str(layer_index)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    print(heatmap.shape)
    heatmap_shape = heatmap.shape[1]
    heatmap = heatmap.reshape((heatmap_shape, heatmap_shape))
    #print('heatmap',heatmap.shape)

    img = cv2.imread(image_directory)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    #print('heatmap',heatmap.shape)
    overlapped_img = heatmap * intensity + img
    #print('overlapped_img', overlapped_img.shape)

    stacked_output_image = np.vstack([img, heatmap, overlapped_img])
    stacked_output_image1 = np.vstack([img, heatmap, overlapped_img])
    print('stacked_output_image1',stacked_output_image1.shape)

    ############    Testing code
    #img3435 = Image.fromarray((stacked_output_image * 255).astype(np.uint8))
    #img3435 = ImageTk.PhotoImage(img3435)
    #print(type(stacked_output_image))
    #print(stacked_output_image.shape)

    ###########  Add layer name to image - Style 1
    '''y = round(0.666 *stacked_output_image.shape[0]) #390
    height = 15
    cv2.rectangle(stacked_output_image, (0,y), (130, y+ height), (0, 0, 0), -1)
    cv2.putText(stacked_output_image, label, (0, y+ height-5), cv2.FONT_HERSHEY_SIMPLEX,	0.3 , (255, 255, 255), 1)'''
    ###########   Add layer name to image -  Style 2
    y = round(stacked_output_image.shape[0])  # 390
    height = 15
    cv2.rectangle(stacked_output_image, (0, y - height), (160, y), (0, 0, 0), -1)
    cv2.putText(stacked_output_image, label, (0, y - height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    res = gradcam_save_res
    # google_colab
    # cv2_imshow(stacked_output_image)
    # cv2_imshow(cv2.resize(cv2.imread(image_directory), (res, res)))

    # my_laptop
    resized_overlapped_img = cv2.resize(overlapped_img, (res, res))
    resized_stacked_output_image = cv2.resize(stacked_output_image, (res, res))

    cv2.imwrite('saved_results/' + image_directory + '_overlapped_' + str(time.time()) + '.png', resized_overlapped_img)
    cv2.imwrite('saved_results/' + image_directory + '_Stacked_' + str(time.time()) + '.png', stacked_output_image)
    # cv2.imshow('amrit', resized_overlapped_img)

    # cv2.imshow('amrit', resized_stacked_output_image)

    return (stacked_output_image1 , resized_overlapped_img , results)

# for layer_index in layer_index_list:
#gradCAM("img2.jpg", layer_index, img_size , model_architecture ,intensity=0.9, gradcam_save_res=250)


