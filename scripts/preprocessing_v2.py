import os
import pickle
import pandas as pd
import tensorflow as tf
from PIL import Image
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
import cv2

def photo_to_array(df,data_directory) :
    dataset = []
    for index, row in df.iterrows():
        image_path = os.path.join(data_directory, row['filename'])
        img = Image.open(image_path)
        img_array = np.array(img)
        dataset.append(img_array)
    dataset = np.array(dataset)
    return dataset



def detect_cars(image_path):
   
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

    # Reading the image
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
               
            else:
                print(f"Invalid classId: {classId}")
               
    largest_rectangle = None
    largest_size = 0

    for rectangle in car_rectangles:
        x, y, width, height = rectangle
        size = width * height

        if size > largest_size:
            largest_size = size
            largest_rectangle = rectangle

    if largest_rectangle is not None:
        x, y, width, height = largest_rectangle

        # Crop the image using the largest rectangle coordinates
        cropped_img = img[y:y+height, x:x+width]

        # Display the cropped image
        cv2.imshow('Cropped Image', cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No rectangles found.")

    return cropped_img


def grayscale(images):
    """
    Transforme une liste d'images RGB en niveaux de gris.
    
    Args:
        images (numpy.ndarray): Tableau 4D représentant une liste d'images RGB.
                               La forme du tableau doit être (nb_images, hauteur, largeur, 3).
    
    Returns:
        numpy.ndarray: Tableau 3D représentant la liste d'images en niveaux de gris.
                       La forme du tableau est (nb_images, hauteur, largeur).
    """
    gray_images = np.dot(images[...,:3], [0.2989, 0.5870, 0.1140])
    return gray_images


def normalize(images):
    """
    Normalise les valeurs d'une liste d'images en les divisant par 255.
    
    Args:
        images (numpy.ndarray): Tableau représentant une liste d'images.
                               La forme du tableau peut être (nb_images, hauteur, largeur)
                               ou (nb_images, hauteur, largeur, 3) pour des images RGB.
    
    Returns:
        numpy.ndarray: Tableau avec les mêmes dimensions que l'entrée, mais avec des valeurs normalisées entre 0 et 1.
    """
    return images / 255.0

def reshape_inter_cubic (width,height,image) :
    img_resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_CUBIC)
    return img_resized

def reshape_inter_linear (width,height,image) :
    img_resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_LINEAR)
    return img_resized

def reshape_inter_nearest (width,height,image) :
    img_resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_NEAREST)
    return img_resized


def pad_image(image, target_shape, pad_value=1000):
    """
    Ajoute un remplissage autour d'une image pour atteindre une forme souhaitée.
    
    Args:
        image (numpy.ndarray): Tableau représentant une image.
                              La forme du tableau peut être (hauteur, largeur)
                              ou (hauteur, largeur, profondeur) pour des images en couleur.
        target_shape (tuple): Tuple contenant la nouvelle forme souhaitée (hauteur, largeur).
        pad_value (float, optional): Valeur utilisée pour le remplissage. Défaut : 1e6 (très grande valeur).
    
    Returns:
        numpy.ndarray: Tableau représentant l'image avec le remplissage ajouté.
    """
    image_shape = image.shape
    
    # Calculer les dimensions du remplissage nécessaire
    height_pad = target_shape[0] - image_shape[0]
    width_pad = target_shape[1] - image_shape[1]
    
    # Calculer le remplissage supérieur et inférieur
    top_pad = height_pad // 2
    bottom_pad = height_pad - top_pad
    
    # Calculer le remplissage gauche et droit
    left_pad = width_pad // 2
    right_pad = width_pad - left_pad
    
    # Ajouter le remplissage
    if len(image_shape) == 2:  # Image en niveaux de gris
        padded_image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=pad_value)
    else:  # Image en couleur
        padded_image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=pad_value)
    
    return padded_image

if __name__=='__main__' :
    image_path = r"C:\Users\remip\Python\Cours S2 2023-2024\albert_project_PoC_G3\cars_project"
    img = detect_cars(image_path)
    img

    data_dir = r'C:\Users\remip\Python\Cours S2 2023-2024\albert_project_PoC_G3\cars_project'
    # img_width, img_height = 480, 360
    filenames = os.listdir(data_dir)
    labels = [filename.split('-')[:2] for filename in filenames]
    labels = pd.DataFrame(labels, columns=['brand', 'model'])
    labels['filename'] = filenames
    labels = labels[labels['model']!='None']

    data_dir = r'C:\Users\remip\Python\Cours S2 2023-2024\albert_project_PoC_G3\cars_project'

    # Créez ds en tant que tableau numpy vide
    ds = np.empty((0,), dtype=object)

    # Itérez sur chaque valeur de la colonne file_path du dataframe labels
    for file_path in labels['file_path']:
        # Construisez le chemin complet de l'image
        image_path = os.path.join(data_dir, file_path)
        
        # Appelez detect_cars avec le chemin complet de l'image
        result = detect_cars(image_path)
        
        # Ajoutez le résultat à ds
        ds = np.append(ds, result)