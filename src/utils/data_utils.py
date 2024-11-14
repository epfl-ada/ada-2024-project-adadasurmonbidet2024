"""
Script containing functions used in data cleaning processes.
"""

import json
import pandas as pd
import nltk
from nltk.corpus import names
from nltk.corpus import words

# Download the necessary nltk datasets (only needed once)
nltk.download('words')
nltk.download('names')

# Load the set of English words and names
english_words = set(words.words())
all_names = set(names.words())

# Define invalid characters and word lists for filtering
invalid_chars = set("-/.,'\"#()0123456789")
invalid_word_list = {
    'american', 'british', 'french', 'italian', 'german', 'spanish', 
    'mexican', 'canadian', 'australian', 'japanese', 'russian', 'chinese', 
    'indian', 'brazilian', 'south korean', 'swedish', 'dutch', 'polish', 
    'irish', 'argentine', 'greek', 'egyptian', 'turkish', 'israeli', 
    'south african', 'nigerian', 'filipino', 'indonesian', 'pakistani', 
    'thai', 'european', 'asian', 'african', 'le', 'mom', 'la'
}
vowels = set('aeiouy')

## Functions

def str_dict_to_values(dict_in_str: str) -> list[str]:
    """
    Convert a string representation of a dictionary to a list of its values.
    """
    if dict_in_str is None:
        return []
    dict_ = json.loads(dict_in_str)
    return list(dict_.values())

def remove_nan_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove rows with NaN values in a specified column.
    """
    return df.dropna(subset=[column])

def filter_non_english_names(name):
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

def keep_names(name: str) -> str:
    """
    Keep only names that are recognized as valid first names (either male or female).
    """
    name_parts = name.split()
    filtered_names = [
        name for name in name_parts 
        if name in all_names 
    ]
    return ' '.join(filtered_names)

def keep_first_name(name: str) -> str:
    """
    Keep only the first name from a full name string.
    """
    return name.split()[0] if name else name
