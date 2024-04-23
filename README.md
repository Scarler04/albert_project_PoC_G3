# Car model identification

## Overview
This project's goal is to create Proof of Concept for a model that can differentiate different car models. This project includes object detection, preprocessing techniques like grayscale, normalization and KMeans, as well as computer vision models such as Convolutional Neural Network (CNN) or Support Vector Machine (SVM)(Still in developement).

## Features
- Feature 1: Utilize computer vision techniques and machine learning algorithms to accurately identify car models depicted in photos. Users can quickly and accurately determine the model of cars in images, aiding in tasks such as cataloging, inventory management, and automotive research.
- Feature 2: This project is only a Proof of Concept, and is very scalable as this PoC is only focused on 15 car models. It could be reproduced on a larger catalogue of car models and eventually go further by indetnifying the producer of the car or the year of production.
- Feature 3: Compare computer vision models and prepocessing methods on a specific case requiring a precise analysis of an image.

## Installation

```bash
git clone https://github.com/Scarler04/albert_project_PoC_G3
cd albert_project_PoC_G3
pip install -r requirements.txt
python .\scripts\CNN.py
python .\scripts\main.py
```
(This code will create a local file with the scraped data in the cloned repository)

## Python Files
[main.py](scripts/main.py) : Main program which regroups the whole optimized process (processes and models that were used in tests but weren't performant enough are not present in main.py)

[scraping.py](scripts/scraping.py) : Regroups the functions used in the scraping of ParuVendu

[preprocessing.py](scripts/preprocessing.py) : Regroups the functions used in the different preprocessing steps. The data is already clean, thus preprocessing here is for optimizing the computing time. Not all are used in the final version of main.py.

[CNN.py](scripts/CNN.py) : Trains the CNN model

## Dataset
The dataset is a set of photos scraped on ParuVendu. You can find the photos on this google drive :
https://drive.google.com/drive/folders/13sQpVtFgQOGI02P-FGD4rb7x7tvAaKF8?usp=sharing

Photo example :
![Car photo](citroen-c3-1.jpg)


