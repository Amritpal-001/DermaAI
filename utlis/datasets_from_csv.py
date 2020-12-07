
'''IMG_SIZE = (64,64)
batch_size = 128
data_directory = './data'
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint , EarlyStopping

import matplotlib.pyplot as plt

### https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_dataframe

import cv2
import numpy as np
from PIL import Image


def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

#Use Dataset.map to create a dataset of image, label pairs:

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())


def custom_function(image):
    image = np.array(image)
    hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    return Image.fromarray(hsv_image)

def generate_dataset_from_dataframe(csv_name , csv_directory , img_size , batch_size):
    datagen = ImageDataGenerator(rescale=0 / 1, rotation_range=10, width_shift_range=0.1, zoom_range=0.1,
                                       horizontal_flip=True,fill_mode='nearest', validation_split=0.2)

    train_gen = datagen.flow_from_dataframe(directory=data_directory, target_size=(img_size , img_size),
                                                  batch_size=batch_size, shuffle=True, subset='training')

    val_gen = datagen.flow_from_dataframe(directory=data_directory, target_size=(img_size , img_size),
                                                batch_size=batch_size, shuffle=True, subset='validation')
    return(train_gen , val_gen)


def compare_model_param_from_dataframe(model, data_directory,epoch ,  img_size, batch_size , learning_rate, patience):
    Adamoptimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=Adamoptimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    train_gen, val_gen = generate_dataset_from_directory(data_directory, img_size, batch_size)
    test_steps_per_epoch = np.math.ceil(train_gen.samples / train_gen.batch_size)
    valid_steps_per_epoch = np.math.ceil(val_gen.samples / val_gen.batch_size)

    save_checkpoint = ModelCheckpoint("SavedModel.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', save_freq=1)
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=patience, verbose=1, mode='auto')

    History = model.fit_generator(train_gen, validation_data=val_gen, steps_per_epoch=test_steps_per_epoch,
                                  validation_steps=valid_steps_per_epoch,
                                  #steps_per_epoch=5, validation_steps = 5,
                                  epochs= epoch, callbacks=(early_stop))

    plt.plot(History.history["accuracy"] , label = 'train')
    plt.plot(History.history['val_accuracy'], label = 'test')
    #plt.plot(History.history['loss'])
    #plt.plot(History.history['val_loss'])
    plt.title('LR='+str(learning_rate) , pad = 40)
    plt.ylabel('acc' + str(round(History.history['accuracy'][-1] , 2)))
    plt.xlabel('val_acc' + str(round(History.history['val_accuracy'][-1] , 2)))

    #return(History)
