import pycountry_convert as pc
import pycountry
import pandas as pd
import country_converter as coco
from googletrans import Translator
from tqdm import tqdm
from google.cloud import translate_v2 as translate
from google.api_core.exceptions import BadRequest
import numpy as np
import sys
from geopy.geocoders import GoogleV3
from tqdm import tqdm

tqdm.pandas()
# tqdm_notebook.pandas()
translator = Translator()
translate_client = translate.Client()
# Change API key below
API_KEY = '#####'
geolocator = GoogleV3(API_KEY)


# print(translate_client.translate(['Brasil', 'Australia'], target_language='en')['translated_text'])

median_income_dict = {
    'AF': 1930,
    'AS': 7350,
    'OC': 53220,
    'EU': 29410,
    'NA': 49240,
    'SA': 8560
}
metadata = pd.read_csv('./data/imagenet/metadata.csv', header=None, encoding='utf-8')

count = 0
metadata[7] = np.nan
country_name_english = None
# print(metadata.head())
for index, row in tqdm(metadata.iterrows()):
    try:
        # print(row)
        country_name_english = translate_client.translate(row[4])
        country_name_english = country_name_english['translatedText']
        country_code = pc.country_name_to_country_alpha2(country_name_english, cn_name_format='default')
        continent_name = pc.country_alpha2_to_continent_code(country_code)
        metadata.iloc[index, 7] = median_income_dict[continent_name]
    except (KeyError, TypeError, BadRequest) as e:

        if country_name_english == 'Korea' or country_name_english == 'Saudi' \
                or country_name_english == 'UAE' or country_name_english == 'Amman'\
                or country_name_english == 'Diameter':
            metadata.iloc[index, 7] = median_income_dict['AS']
        elif country_name_english == 'The Bahamas' or country_name_english == 'The Savior':
            metadata.iloc[index, 7] = median_income_dict['NA']
        elif country_name_english == 'mali' or country_name_english == 'lesotho':
            metadata.iloc[index, 7] = median_income_dict['AF']
        elif country_name_english == 'Vatican CITY':
            metadata.iloc[index, 7] = median_income_dict['EU']
        else:
            print(e)
            print(index, country_name_english)
        metadata.to_pickle('./data/imagenet/metadata_with_income_2.pkl')
        continue

metadata.to_pickle('./data/imagenet/metadata_with_income_2.pkl')