import re
import time
import os
import cv2

import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#read data
PATH = "./data/static_html/"
directory = os.fsencode(PATH)
image_data = []
label_geo_data = []
label_dict = {}
label_index = 0

for file in os.listdir(directory):
    text_filename = os.fsencode(file)
    text_filename = text_filename.decode("utf-8")
    print("On filename: {}".format(text_filename))
    label_dict[text_filename[:-5]] = label_index

    if text_filename.endswith(".html"):
        FILE_PATH = PATH + text_filename
        with open(FILE_PATH, 'r', encoding='utf-8') as fp:
            data = fp.read().replace('\n', '')

        #apply beautifulsoup
        soup = BeautifulSoup(data, "html.parser")
        matrix = soup.find("matrix-images").find_all("div", {"class": "cell-inner"})

        try:
            for elem in tqdm(matrix):
                #get required attributes
                image_link = "https:" + re.findall("url\\(\"(.+)\"\\);", elem['style'])[0]
                income = re.sub("[$ ]", "", elem.find("span", {"class": "place-image-box-income"}).text)
                country =  elem.find("span", {"class": "place-image-box-country"}).text
                response = requests.get(image_link)
                if response.status_code == 200:
                    with open('temp.jpg', 'wb') as f:
                        f.write(response.content)
                    resized_image = cv2.resize(cv2.imread('temp.jpg'), (224, 224))
                    resized_image_rgb = (cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
                    image_data.append(resized_image_rgb)
                    label_geo_data.append(np.array([label_index, country, int(income)]))
            label_index += 1
        except KeyError:
            continue


np.save('./data/image_data.npy', np.array(image_data))
np.save('./data/label_geo_data.npy', np.array(label_geo_data))
np.save('./data/label_dict.npy', label_dict)
