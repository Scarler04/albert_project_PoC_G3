import os
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set the directory and parameters
data_dir = r'C:\Users\user\Documents\cars_project'
img_width, img_height = 480, 360
batch_size = 32
epochs = 35

# Load filenames and extract labels
filenames = os.listdir(data_dir)
labels = [filename.split('-')[:2] for filename in filenames]
labels = pd.DataFrame(labels, columns=['brand', 'model'])
labels['filename'] = filenames


# Kmeans 

k = 15
kmeans = KMeans(n_clusters = k, random_state=42)
kmeans.fit(filenames)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

#End of Kmeans