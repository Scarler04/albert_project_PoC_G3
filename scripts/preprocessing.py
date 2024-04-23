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

# Object Detection

import cv2

thres = 0.50  # Threshold to detect objects

# Load class names from coco.names file
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the pre-trained model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def detect_cars(image_path):
    img = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if img is None:
        print("Error: Image not found or invalid image path.")
        return [], None

    img = cv2.resize(img, (700, 500))

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    car_rectangles = []

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Ensure classId is within the valid range and corresponds to a car
            if 0 <= classId < len(classNames) and classNames[classId - 1].lower() == 'car':
                car_rectangles.append(box)
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                print(f"Invalid classId: {classId}")

    return car_rectangles, img

# Replace 'path_to_your_image.jpg' with the actual path to the image you want to detect cars in
image_path = r"C:\Users\user\Documents\cars_project\peugeot-2008-8.jpg"
car_rectangles, img = detect_cars(image_path)
print("Car rectangles:", car_rectangles)

# Display the image with detected objects
cv2.imshow('Image with Detected Cars', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
