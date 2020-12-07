

# import the necessary packages
from utlis.gradcam import GradCAM
import tensorflow
#from tensorflow.keras.applications import ResNet50
#from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2

physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

img = 'img.jpg'

#Model = ResNet50
#Model = VGG16

#model = Model(weights="imagenet")
#model = VGG16(weights="imagenet")
model = tensorflow.keras.models.load_model('/home/amritpal/PycharmProjects/100-days-of-code/100_days_of_code/Skin_disease_working_version_minimal/saved_model/model_name_RGB_128__0.19__5.395__12-2_22:56')

orig = cv2.imread(img)
resized = cv2.resize(orig, (224, 224))
image = load_img(img, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

preds = model.predict(image)
i = np.argmax(preds[0])

'''decoded = imagenet_utils.decode_predictions(preds)
(imagenetID, label, prob) = decoded[0][0]
label = "{}: {:.2f}%".format(label, prob * 100)
print("[INFO] {}".format(label))'''

label = 'amrit'


# initialize our gradient class activation map and build the heatmap
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image)
# resize the resulting heatmap to the original input image dimensions
# and then overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

#del history
del model
#gc.collect()


# draw the predicted label on the output image
cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.8, (255, 255, 255), 2)
# display the original image and resulting heatmap and output image
# to our screen
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)
cv2.imshow("Output", output)
cv2.waitKey(0)


#python3 apply_gradcam.py --image img.jpg
