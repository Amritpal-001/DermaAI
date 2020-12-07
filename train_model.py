

import tensorflow as tf
import numpy as np

#from tensorflow.keras.applications.resnet import ResNet50 #ResNet152 ResNet101

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

#tf.config.list_physical_devices('GPU')
from model.architecture import *
from utlis.datasets  import *
from utlis.save_results_temp  import *
from model.inception_v3_architecture import *

from init_variables import *

def set_random_seed(seed_value):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print('Random_seed set for' , seed_value)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
set_random_seed(seed_value)


try:
    del model
    print('deleting old model')
except:
    pass

#################model = custom_original_InceptionV3_base(num_classes = num_classes , input_shape= (299 , 299 , 3))

#from tensorflow.keras.utils import plot_model
input_size  = 224
batch_size = 4
model = custom_original_InceptionV3_base(num_classes, input_shape=(input_size, input_size, 3))
#model.summary()
Freeze_model(model)
#model.summary()
#plot_model(model, to_file='model.png' , show_shapes=False,)
#plot_model(model, to_file='model.png' , show_shapes=False,)

'''directory = '/home/amritpal/PycharmProjects/100-days-of-code/100_days_of_code/Skin_lesions_Classification-master/isic_2020/train/'

df = pd.read_csv(directory + 'train.csv')

# Reading files from path in data frame
datagen = ImageDataGenerator(rescale=1.0 / 255.0, rotation_range=10, width_shift_range=0.1, zoom_range=0.1,
                             horizontal_flip=True, fill_mode='nearest', validation_split=0.2)

train_gen = datagen.flow_from_dataframe(df, directory=directory, x_col='image_name', y_col='label', class_mode='raw',
                                        subset='training')
val_gen = datagen.flow_from_dataframe(df, directory=directory, x_col='image_name', y_col='label', class_mode='raw',
                                      subset='validation')'''

#input_shape = (64,64,3)
#model = CBR_test(input_shape,num_classes)

train_test_dataframe_model(model, epoch, learning_rate)

model_name = gen_model_name('model_name', batch_size, learning_rate)
model_name = model_name + str(time.time())

del model

plt.savefig(save_location + '/' + model_name + '.png')
plt.show()

tf.keras.backend.clear_session()



'''for i in range(0, 3):
    plot_no = 420 + (i + 1)
    plt.subplot(plot_no )
    if i == 0:

    if i == 1:
        model = custom_InceptionV3_base(num_classes, input_shape=(img_size, img_size, 3))

    train_test_model(model, data_directory, epoch, rescale, img_size, batch_size, learning_rate, patience,
                     validation_split)
    model_name = gen_model_name('model_name', batch_size, learning_rate)
    model_name = model_name + str(time.time())

plt.savefig(save_location + '/' + model_name + '.png')
plt.show()'''

