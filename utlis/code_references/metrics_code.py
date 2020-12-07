


from tensorflow.keras.callbacks import Callback
import numpy as np
from sklearn.metrics import balanced_accuracy_score

SGDoptimizer = tensorflow.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
#momentum: float >= 0
Adamoptimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
Nadamoptiizer = tensorflow.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
RMSpropoptimizer = tensorflow.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

#loss = 'mean_absolute_error''categorical_crossentropy'



