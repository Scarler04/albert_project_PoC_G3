* Creating a clear and informative README file is crucial for any project, as it often serves as the first point of reference for anyone who encounters your project repository.
* Below is a template to start with: 

# Project Title

Car model identification

## Overview
This project's goal is to create Proof of Concept for a model that can differentiate different car models. This projects includes object detection, preprocessing techniques like grayscale, bormalization and KMeans, as well as computer vision models such as Convolutional Neural Network (CNN) or Support Vector Machine (SVM)(Still in developement).

## Features
- Feature 1: Utilize computer vision techniques and machine learning algorithms to accurately identify car models depicted in photos. Users can quickly and accurately determine the model of cars in images, aiding in tasks such as cataloging, inventory management, and automotive research.
- Feature 2: This project is only a Proof of Concept, and is very scalable as this PoC is only focused on 15 car models. It could be reproduced on a larger catalogue of car models and eventually go further by indetnifying the producer of the car or the year of production.
- Feature 3: Compare computer vision models and dimensionality reduction methods on a specific case requiring a precise analysis of an image.

## Installation
Clone the repository.

Launch [CNN.py](notebooks/CNN.py).

In main.py, change file path line xx to the same path as the cloned repository.

Launch [main.py](notebooks/main.py).

## Python Files
[main.py](notebooks/main.py) : Main program which regroups the whole optimized process (processes and models that were used in tests but weren't performant enough are not present in main.py)

[scraping.py](notebooks/scraping.py) : Regroups the functions used in the scraping of ParuVendu

[preprocessing.py](notebooks/preprocessing.py) : Regroups the functions used in the different preprocessing steps. The data is already clean, thus preprocessing here is for optimizing the computing time. Not all are used in the final version of main.py.

[CNN.py](notebooks/CNN.py) : Trains the CNN model

## Dataset
The dataset is a set of photos scraped on ParuVendu. You can find the photos on this google drive :
https://drive.google.com/drive/folders/13sQpVtFgQOGI02P-FGD4rb7x7tvAaKF8?usp=sharing

