import selenium
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By  
import pandas as pd
from datetime import datetime
import xlsxwriter
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import os
import sys
import requests

def extract_image_links(url):
    
    parent = driver.find_element(By.CLASS_NAME, "flex.sm\\:block.gap-4")
    child_element = parent.find_element(By.TAG_NAME, "a")
    try:
        link = child_element.get_attribute("href")
    except:
        link = "a/A/A/A/NONE/NONE/NONE/NONE"
    brand, model = link.split('/')[5:7]
    a = brand
    b = model
    print(a, b)
    
    truc = []
    c = 0
    while c<5:
        photos = driver.find_elements(By.CLASS_NAME, "img.basis-80.w-80.sm\\:w-full.h-50.text-center.relative.loading")
        print(len(photos))

        for photo in photos:
            phot = photo.find_element(By.CLASS_NAME, "w-full.object-cover.h-50")
            phot = phot.get_attribute("src")
            truc.append(phot)
        print("got links")
        pgsv = driver.find_element(By.CLASS_NAME, "pgsuiv")
        pgsv = pgsv.find_element(By.TAG_NAME, "a")
        print("got next page")
        try:
            element = driver.find_element(By.CSS_SELECTOR, "a.relative.inline-flex.items-center.border.border-gray-300.bg-white.px-3.5.py-3.5.text-sm.font-medium.hover:bg-gray-50.focus:z-20.cursor-pointer")
            driver.execute_script("arguments[0].click();", element)
        except:
            nv_url = url + "?p=" + str(c+1)
            driver.get(nv_url)
        print("next page")
        c+=1
            
    i=0
    for link in truc:
        # Téléchargez et enregistrez chaque image dans le dossier "car_images"
        file_name = f"{a}-{b}-{i+1}.jpg"
        save_path = os.path.join("cars_project", file_name)
        download_image(link, save_path)
        sys.stdout.write(f"\rDownloaded {i+1}/{len(truc)} images")
        i+=1
    print("\nDone!")
    return truc

def download_image(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
    else:
        print(f"Error {response.status_code}: unable to download image from {url}")


