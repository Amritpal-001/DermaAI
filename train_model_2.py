

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


'''try:
    print('deleting old model')
    del model
except:
    pass'''

#################model = custom_original_InceptionV3_base(num_classes = num_classes , input_shape= (299 , 299 , 3))

#from tensorflow.keras.utils import plot_model

fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(20)


model = custom_original_InceptionV3_base(num_classes, input_shape=(img_size, img_size, 3))
#model.summary()
Freeze_model(model)
#model.summary()
#plot_model(model, to_file='model.png' , show_shapes=False,)
#plot_model(model, to_file='model.png' , show_shapes=False,)

train_test_model(model, data_directory, epoch, img_size, batch_size, learning_rate, patience,validation_split)

model_name = gen_model_name('model_name', batch_size, learning_rate)
model_name = model_name + str(time.time())

plt.savefig(save_location + '/' + model_name + '.png')
plt.show()


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

