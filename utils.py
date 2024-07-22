import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

# function to correctly read the data from the csv
def load_data():
    # read the data from the csv
    data = pd.read_csv('./data/train.csv')

    # convert the pandas to numpy
    data = np.array(data)

    # seperate the input and output
    y = data[:, 0]
    x = data[:, 1:]

    #split the data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    return X_train, X_test, Y_train, Y_test
    
# defining activation functions
def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Define a function to load the model parameters
def load_params(filename_prefix):
    W1 = np.load(f'{filename_prefix}_W1.npy')
    b1 = np.load(f'{filename_prefix}_b1.npy')
    W2 = np.load(f'{filename_prefix}_W2.npy')
    b2 = np.load(f'{filename_prefix}_b2.npy')
    W3 = np.load(f'{filename_prefix}_W3.npy')
    b3 = np.load(f'{filename_prefix}_b3.npy')
    return W1, b1, W2, b2, W3, b3

# defining derivative functions
def deriv_ReLU(Z):
    return Z > 0

# coverted into one hot
def label_conversion(y):
    y_converted = np.zeros((y.size, y.max() + 1))
    y_converted[np.arange(y.size), y] = 1
    return y_converted

# conver user label to softmax
def user_label(y):
    y_converted = np.zeros((1, 10))
    y_converted[0, y] = 1
    return y_converted

# defining forward propagation
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = np.dot(X, W1) + b1 
    A1 = ReLU(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = ReLU(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# defining backwards propagation
def backward_prop(Z1, Z2, A1, A2, A3, W3, W2, X, Y):
    y_labelled = user_label(Y)

    dZ3 = A3 - y_labelled
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dZ2 = np.dot(dZ3, W3.T) * deriv_ReLU(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, W2.T) * deriv_ReLU(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3

# defining a function to update the parameters
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1

    W2 -= alpha * dW2
    b2 -= alpha * db2

    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3

# defining prediction
def make_prediction(W1, b1, W2, b2, W3, b3, X):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    return np.argmax(A3, axis=1)

# function to process an image to vector
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # if the image doesn't load then show an error
    if img is None:
        raise ValueError("Image not loaded. Check the path.")
    img = cv2.resize(img, (28, 28))
    img_vector = img.flatten()
    return img_vector / 255.0

# Function to plot the image and display the prediction
def plot_image_with_prediction(prediction):
    img = cv2.imread('test_image.png')
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted Number: {prediction}')
    plt.axis('off')
    plt.show()

# function to raise warning
def display_error_and_wait(error_message):
    print(f"\033[91m{error_message}\033[0m\n")  # Print in red

# function to learn from mistake
def learn_from_input(W1, b1, W2, b2, W3, b3, X, Y, alpha):
    print('Learning...')
    Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, Z2, A1, A2, A3, W3, W2, X, Y)
    W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
    return W1, b1, W2, b2, W3, b3