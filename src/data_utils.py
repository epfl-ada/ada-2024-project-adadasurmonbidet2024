" Script with all functions used in data cleaning processes "
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import geopandas as gpd
import data_utils
import nltk
from nltk.corpus import words
from nltk.corpus import names


## Convert a string representation of a dictionary to a list of its values.
def str_dict_to_values(dict_in_str: str)->list[str]:
    if dict_in_str is None:  
        return []
    dict_ = json.loads(dict_in_str)
    values = list(dict_.values())
    return values

## Remove NaN rows
def remove_nan_rows(df:pd.DataFrame,column:str)->pd.DataFrame:
    no_nan_df = df.dropna(subset=[column])
    return no_nan_df

## Filters out non-English words from a name based on predefined criteria
# Load the set of English words
english_words = set(words.words())

# Define invalid characters and invalid word list
invalid_chars = ['-', '/', '.', ',', "'", '"', '#', "(", ')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
invalid_word_list = [
    'american', 'british', 'french', 'italian', 'german', 'spanish', 
    'mexican', 'canadian', 'australian', 'japanese', 'russian', 'chinese', 
    'indian', 'brazilian', 'south korean', 'swedish', 'dutch', 'polish', 
    'irish', 'argentine', 'greek', 'egyptian', 'turkish', 'israeli', 
    'south african', 'nigerian', 'filipino', 'indonesian', 'pakistani', 'thai',
    'european', 'asian', 'african', 'le', 'mom', 'la'
]
# Define the set of vowels
vowels = set('aeiouy')

# Define the filtering function
def filter_non_english_names(name: str) -> str:
    words_in_name = name.split()
    filtered_words = [
        word for word in words_in_name 
        if word.lower() not in english_words 
        and word.lower() not in invalid_word_list
        and all(char not in word for char in invalid_chars) 
        and any(char in vowels for char in word.lower())
        and sum(1 for char in word if char.isupper()) <= 1
    ]
    return ' '.join(filtered_words)

## Keeping only the names of a dataset
# Download the names dataset (only need to do this once)
nltk.download('names')

# Load all names (both male and female) from nltk
all_names = set(names.words())

# Function to check if the name is a first name (male or female)
def keep_names(name):

    name_parts = name.split()
    filtered_names = [
        name for name in name_parts 
        if name in all_names 
    ]
    return ' '.join(filtered_names)

## Keep only first names
def keep_first_name(name):
    names = name.split()
    if (len(names)>1):
        name = names[0]
    else:
        name = name
    return name
