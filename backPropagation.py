import numpy as np

# step -1 define the activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#step-2 for the purpose of weight adjustment, we perform back propagation
def sigmoid_derivative(x):
    return x*(1-x)

#step -3 Training data (xs: inputs, ys: expected output)
xs = np.array([[0,0],[0,1],[1,0],[1,1]])
ys = np.array([[0],[1],[1],[0]])

#step-4 Initilaizing the weights
np.random.seed(1)
weights = np.random.random((2,1))
# bias = np.random.rand(1)

#step-5 initilaising learning rate
learning_rate = 0.1
epochs = 10

for epoch in range(epochs):
  #forward propagation aka feed forwarding
  input_layer = xs
  outputs = sigmoid(np.dot(input_layer, weights)) # dot does the function of multiplication that we do like (w12 * x1) + (w21 * x2)

  #calculate the error as a differnce between observed and the expected
  error = ys - outputs

  #perform back propagation to adjust the errors
  adjustments = error * sigmoid_derivative(outputs)
  weights += np.dot(input_layer.T, adjustments)

  # back propagation
  error = ys - outputs
  adjustments = error * sigmoid_derivative(outputs)
  weights += np.dot(input_layer.T, adjustments) * learning_rate # .T is transpose to convert input layer that is of 1d array into 2d array

#step - 6
print("Weights after training")
print(weights)

print("Outputs after training")
print(outputs)

#plotting the graph for this
import matplotlib.pyplot as plt
plt.plot(outputs)
plt.ylabel('outputs')
plt.show()
