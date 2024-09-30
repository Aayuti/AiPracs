try:
  import tensorflow.compat.v2 as tf
  #preferred for numerical computation and large scale ML;
  #open source, created by google team

except Exception:
  pass

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
#open source software library that provides a python interface for ai neural net, acts as a tensorflow lib, used to evaluate deep learning models

import tensorflow as tf
from keras import Sequential
# Sequential is a class in Keras which helps in building model
# constructor of sequential returns a final model
# training and inference features on this model

from keras.layers import Dense # dense is a class in keras, they act like bricks to build the model

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) # 1 unit means 1 input, input_shape determines the size.
#argument for input shape can we re written as Input(shape=(3,))

#Dense implements the operation:
#output = activation(dot(input, kernel)+bias)

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy']) # sgd is stochastic gradient descent
# bowl example - we start from brim and reach the bottom for best/ optimal solution
# descent means going down
# stochastic means randomly we are going down, that is probably in zig zag also
# metrics parameter is for viewing the data

# optimizer finds a solution with the lowest cost,
# fast execution, lowest memory has differeny=t gradients
# sgd = stochastic gradient descent
# other opimizers: Adam, RMSprop, etc
# loss function tells us by how much are we missing the mark

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

history = model.fit(xs, ys, epochs=20)
# epochs is one entire iteration from forward to back
# for the model to get trained; asssuming 500 is sufficient
# for the machine to learn

prediction_input = np.array([15.0])  #converts list to array
print(model.predict(prediction_input))


plt.plot(history.history['accuracy'], label='model accuracy', color='red')
plt.plot(history.history['loss'], label='model loss', color='blue')
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
plt.legend()
plt.show()



