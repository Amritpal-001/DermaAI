import tensorflow as tf
from tensorflow.python.client import device_lib


#tf.debugging.set_log_device_placement(True)
tf.test.is_gpu_available()
#print(device_lib.list_local_devices())


tf.config.list_physical_devices('GPU')
print(device_lib.list_local_devices())

###  https://www.tensorflow.org/install/gpu#linux_setup

### https://stackoverflow.com/questions/55224016/importerror-libcublas-so-10-0-cannot-open-shared-object-file-no-such-file-or


### https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in
