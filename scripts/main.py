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

# Function Imports 
from scraping import extract_image_links
from scraping import download_image


if __name__=='__main__':
        
    # Scraping
    driver = webdriver.Edge()

    liste_url = [
        "https://www.paruvendu.fr/a/voiture-occasion/peugeot/208/",
        "https://www.paruvendu.fr/a/voiture-occasion/renault/clio/",
        "https://www.paruvendu.fr/a/voiture-occasion/dacia/sandero/",
        "https://www.paruvendu.fr/a/voiture-occasion/citroen/c3/",
        "https://www.paruvendu.fr/a/voiture-occasion/peugeot/2008/",
        "https://www.paruvendu.fr/a/voiture-occasion/renault/captur/",
        "https://www.paruvendu.fr/a/voiture-occasion/peugeot/3008/",
        "https://www.paruvendu.fr/a/voiture-occasion/toyota/yaris/",
        "https://www.paruvendu.fr/a/voiture-occasion/citroen/c4/",
        "https://www.paruvendu.fr/a/voiture-occasion/volkswagen/golf/",
        "https://www.paruvendu.fr/a/voiture-occasion/renault/megane/",
        "https://www.paruvendu.fr/a/voiture-occasion/ford/fiesta/",
        "https://www.paruvendu.fr/a/voiture-occasion/peugeot/308/",
        "https://www.paruvendu.fr/a/voiture-occasion/renault/twingo/",
        "https://www.paruvendu.fr/a/voiture-occasion/volkswagen/polo/"
    ]

    nb_model = 1
    for url in liste_url:
        driver.get(url)
        liste = pd.DataFrame()
        print("Driver créé")
        
        if nb_model == 1:
            cookie = driver.find_element(By.XPATH, "//button[@onclick='cmp_pv.cookie.saveConsent(true);']")
            cookie.click()

        os.makedirs("cars_project", exist_ok=True)

        extract_image_links(url)
        nb_model+=1

    driver.quit()