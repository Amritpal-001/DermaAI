

# import the necessary packages
from utlis.gradcam import *
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2
from init_variables import *
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

def Get_predictions_and_GradCam_activation(model_directory , image , show = True, annotate_output = False , label  = None):
    model = tf.keras.models.load_model(model_directory)
    orig = cv2.imread(image)
    #resized = cv2.resize(orig, (224, 224))
    image = load_img(image, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    predictions = model.predict(image)

    i = np.argmax(predictions[0])
    layerName = 'block5_conv3'
    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)
    # resize the resulting heatmap to the original input image dimensions and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    #del history
    del model
    #gc.collect()

    if annotate_output == True:
        # draw the predicted label on the output image
        cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,	0.8, (255, 255, 255), 2)
    # display the original image and resulting heatmap and output image to our screen
    output_image = np.vstack([orig, heatmap, output])
    output_image = imutils.resize(output_image, height=700)

    if show == True:
        cv2.imshow("Output", output_image)
        cv2.waitKey(0)

    return(output_image)




''''image_directory = 'img1.jpg'
model_directory = saved_cbr_weights_directory
#loaded_model = tf.keras.models.load_model(model_directory)
loaded_model = InceptionV3()'''


#for layer in reversed(loaded_model.layers):
#    print(layer.name)
'''b = 0
for n in reversed(loaded_model.layers):
    #layerName = loaded_model.layers[n]
    print(n , b)
    b += 1'''
    
    
'''#print(layerName)
#return(layerName)
orig = cv2.imread(image_directory)
resized = cv2.resize(orig, (224, 224))

image = load_img(image_directory, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)
predictions = loaded_model.predict(image)'''

#Get_predictions_and_GradCam_activation(model_directory , image_directory , show = True)

#orig = 'img1.jpg'

def gradCAM(orig, loaded_model , intensity=0.5, res=250):
    model = loaded_model
    img = image.load_img(orig, target_size=(299, 299))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print(decode_predictions(preds)[0][0][1])  # prints the class of image

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv2d_93')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((8, 8))

    img = cv2.imread(orig)
    img1 = cv2.imread(orig)
    img1 = cv2.resize(img, (orig.shape[1], orig.shape[0]))
    cv2.imshow("Output", img)
    #cv2.waitKey(0)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    cv2.imshow("Output", heatmap)
    #cv2.waitKey(0)


    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cv2.imshow("Output", heatmap)
    #cv2.waitKey(0)

    img_final = heatmap * intensity + img
    img_final = cv2.resize(img_final, (orig.shape[1], orig.shape[0]))
    #cv2.imshow("Output", img_final)
    #cv2.waitKey(0)

    output_image = np.vstack([img_final , heatmap , img1])
    #output_image = imutils.resize(output_image, height=700)

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)

    #cv2_imshow(cv2.resize(cv2.imread(orig), (res, res)))
    #cv2_imshow(cv2.resize(img, (res, res)))

#gradCAM("img1.jpg" , loaded_model)
#gradCAM("Thinking-of-getting-a-cat.png")



def Get_GradCam_activation(loaded_model, image_directory, predictions, annotate_output=False, label=None):
    orig = cv2.imread(image_directory)
    resized = cv2.resize(orig, (224, 224))

    image = load_img(image_directory, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    i = np.argmax(predictions[0])


    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(loaded_model, i )
    heatmap = cam.compute_heatmap(image )
    # resize the resulting heatmap to the original input image dimensions and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    if annotate_output == True:
        # draw the predicted label on the output image
        cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # display the original image and resulting heatmap and output image to our screen
    output_image = np.vstack([orig, heatmap, output])
    output_array = imutils.resize(output_image, height=700)
    print(type(output_array))

    return (output_array)


#Get_GradCam_activation(loaded_model, image_directory, predictions, annotate_output=False, label=None)



'''
amrit conv2d_3
conv2d_3
<class 'numpy.ndarray'>
(700, 310, 3)
<class 'numpy.ndarray'>'''

