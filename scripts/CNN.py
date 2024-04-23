import tensorflow as tf
from tensorflow.keras import datasets,layers, models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


model = models.Sequential([
  layers.Conv2D(32, (3, 3), strides = (1,1), activation='relu', input_shape=(28, 28, 1)),
  layers.MaxPooling2D((2, 2), strides = (1,1)),
  layers.Conv2D(64, (3, 3), strides = (1,1), activation='relu'),
  layers.MaxPooling2D((2, 2), strides = (1,1)),
  layers.Conv2D(64, (3, 3), strides = (1,1), activation='relu'),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(15, activation='softmax')
])





print (model.summary())