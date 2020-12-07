
'''IMG_SIZE = (64,64)
batch_size = 128
data_directory = './data'
'''
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint , EarlyStopping

import matplotlib.pyplot as plt
from init_variables import *
import pandas as pd


#https://machinelearningmastery.com/how-to-evaluate-pixel-scaling-methods-for-image-classification/
##custom image modeification in keras

### https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_dataframe
def generate_dataset_from_dataframe( ):

    #directory   = '/home/amritpal/PycharmProjects/100-days-of-code/100_days_of_code/Skin_lesions_Classification-master/isic_2020/train/'
    directory   = '/content/Skin_lesions_Classification/data/train/'


    df = pd.read_csv(directory + 'train_train.csv')

    # Reading files from path in data frame
    datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=50, width_shift_range=0.3, zoom_range=0.2,
                                       horizontal_flip=True, vertical_flip=True,fill_mode='nearest', validation_split= validation_split, )

                                        
    train_gen = datagen.flow_from_dataframe(df, directory= directory,  x_col='image_name', y_col='label' , class_mode='raw' ,shuffle= True , subset='training')
    val_gen = datagen.flow_from_dataframe(df, directory= directory,  x_col='image_name', y_col='label' , class_mode='raw' , shuffle= True ,subset='validation')

    return(train_gen , val_gen)


def train_test_dataframe_model(model, epoch, learning_rate):
    Adamoptimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=Adamoptimizer, loss='BinaryCrossentropy', metrics=[tf.keras.metrics.mae,  tf.keras.metrics.categorical_accuracy , binary_accuracy(threshold=0.5)])

    train_gen, val_gen = generate_dataset_from_dataframe( )

    test_steps_per_epoch = np.math.ceil(train_gen.samples / train_gen.batch_size)
    valid_steps_per_epoch = np.math.ceil(val_gen.samples / val_gen.batch_size)

    '''save_checkpoint = ModelCheckpoint("SavedModel.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='auto', save_freq=1)
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=patience, verbose=1, mode='auto')'''

    History = model.fit(train_gen, validation_data=val_gen, steps_per_epoch=test_steps_per_epoch,
                                  validation_steps=valid_steps_per_epoch, epochs=epoch , shuffle =True,)

    plt.plot(History.history["accuracy"], label='train accuracy')
    plt.plot(History.history['val_accuracy'], label='Val accuracy')
    plt.plot(History.history['loss'], label='loss')
    plt.plot(History.history['val_loss'], label='val_loss')
    plt.title('LR=' + str(learning_rate), pad=5)
    plt.ylabel('acc' + str(round(History.history['accuracy'][-1], 2)))
    plt.xlabel('val_acc' + str(round(History.history['val_accuracy'][-1], 2)))



'''def generate_dataset_from_directory(data_directory , img_size , batch_size , validation_split ):

    train_datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=50, width_shift_range=0.3, zoom_range=0.2,
                                       horizontal_flip=True, vertical_flip=True,fill_mode='nearest', validation_split= validation_split, )


    train_gen = train_datagen.flow_from_directory(directory=data_directory, target_size=(img_size , img_size),
                                                  batch_size=batch_size, shuffle=True, subset='training')

    val_gen = train_datagen.flow_from_directory(directory=data_directory, target_size=(img_size , img_size),
                                                batch_size=batch_size, shuffle=True, subset='validation')
    return(train_gen , val_gen)


def train_test_model(model, data_directory, epoch, img_size, batch_size, learning_rate, patience, validation_split):
    Adamoptimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=Adamoptimizer, loss='BinaryCrossentropy', metrics=[tf.keras.metrics.mae,  tf.keras.metrics.categorical_accuracy , binary_accuracy(threshold=0.5)])

    train_gen, val_gen = generate_dataset_from_directory(data_directory, img_size, batch_size, validation_split)

    test_steps_per_epoch = np.math.ceil(train_gen.samples / train_gen.batch_size)
    valid_steps_per_epoch = np.math.ceil(val_gen.samples / val_gen.batch_size)

    save_checkpoint = ModelCheckpoint("SavedModel.h5", monitor='val_accuracy', verbose=3, save_best_only=True,
                                      save_weights_only=False, mode='auto', save_freq=1)
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=patience, verbose=1, mode='auto')

    History = model.fit_generator(train_gen, validation_data=val_gen, steps_per_epoch=test_steps_per_epoch,
                                  validation_steps=valid_steps_per_epoch,  epochs=epoch)
                                  

    plt.plot(History.history["accuracy"], label='train accuracy')
    plt.plot(History.history['val_accuracy'], label='Val accuracy')
    plt.plot(History.history['loss'] , label='loss')
    plt.plot(History.history['val_loss'] , label='val_loss')
    plt.title('LR=' + str(learning_rate), pad=5)
    plt.ylabel('acc' + str(round(History.history['accuracy'][-1], 2)))
    plt.xlabel('val_acc' + str(round(History.history['val_accuracy'][-1], 2)))

    # return(History)
'''

'''def generate_dataset_from_directory(data_directory , img_size , batch_size , rescale, rotation_range=rotation_range, width_shift_range=width_shift_range, zoom_range=zoom_range,
                                       horizontal_flip=horizontal_flip , fill_mode=fill_mode, validation_split= validation_split,
                                       featurewise_center=featurewise_center, featurewise_std_normalization=featurewise_std_normalization ):


    train_datagen = ImageDataGenerator(rescale=rescale, rotation_range=10, width_shift_range=0.1, zoom_range=0.1,
                                       horizontal_flip=True,fill_mode='nearest', validation_split= validation_split,
                                       featurewise_center=featurewise_center, featurewise_std_normalization=featurewise_std_normalization )

    train_gen = train_datagen.flow_from_directory(directory=data_directory, target_size=(img_size , img_size),
                                                  batch_size=batch_size, shuffle=True, subset='training')

    val_gen = train_datagen.flow_from_directory(directory=data_directory, target_size=(img_size , img_size),
                                                batch_size=batch_size, shuffle=True, subset='validation')
    return(train_gen , val_gen)'''




'''##IMage from dataset with your own custom functions
def custom_function(image):
    image = np.array(image)
    hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    return Image.fromarray(hsv_image)

def generate_dataset_from_directory_with_custom_function(data_directory , img_size , batch_size , custom_function):
    datagen = ImageDataGenerator(rescale=0 / 1, rotation_range=10, width_shift_range=0.1, zoom_range=0.1,
                                       horizontal_flip=True,fill_mode='nearest', validation_split=0.2 , preprocessing_function = custom_function)

    train_gen = datagen.flow_from_directory(directory=data_directory, target_size=(img_size , img_size),
                                                  batch_size=batch_size, shuffle=True, subset='training')

    val_gen = datagen.flow_from_directory(directory=data_directory, target_size=(img_size , img_size),
                                                batch_size=batch_size, shuffle=True, subset='validation')
    return(train_gen , val_gen)'''

'''def train_model(model, data_directory,epoch ,  img_size, batch_size , learning_rate):
    Adamoptimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=Adamoptimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    train_gen, val_gen = generate_dataset_from_directory(data_directory, img_size, batch_size)
    test_steps_per_epoch = np.math.ceil(train_gen.samples / train_gen.batch_size)
    valid_steps_per_epoch = np.math.ceil(val_gen.samples / val_gen.batch_size)

    save_checkpoint = ModelCheckpoint("SavedModel.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', save_freq=1)
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

    History = model.fit_generator(train_gen, validation_data=val_gen, steps_per_epoch=test_steps_per_epoch,
                                  validation_steps=valid_steps_per_epoch,
                                  #steps_per_epoch=5, validation_steps = 5,
                                  epochs= epoch, callbacks=(early_stop))

    return(History)'''

