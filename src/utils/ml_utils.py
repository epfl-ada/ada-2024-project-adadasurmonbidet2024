"""
Script containing functions used in ML processes.
"""

import pandas as pd
import numpy as np
from jellyfish import soundex
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

def get_vowel_stats(df: pd.DataFrame, category:str) -> tuple:
    vowels = set('aeiouy')

    # Count vowels in a name
    def count_vowels(name):
        return sum(1 for char in name.lower() if char in vowels)

    # Count consonants in a name (alphabetic characters excluding vowels)
    def count_consonants(name):
        return sum(1 for char in name.lower() if char.isalpha() and char not in vowels)

    # Add counts to the DataFrame
    df['vowel_count'] = df[category].apply(count_vowels)
    df['consonant_count'] = df[category].apply(count_consonants)
    df['name_length'] = df['vowel_count'] + df['consonant_count']

def find_unusual_characters(df, column_name, allowed_chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    """
    Identify all unique characters in a column that are not in the allowed characters.

    Parameters:
        df (pd.DataFrame): The dataset containing the column.
        column_name (str): The name of the column to analyze.
        allowed_chars (str): A string of allowed characters.

    Returns:
        set: A set of unique unusual characters.
    """
    allowed_set = set(allowed_chars)
    
    all_characters = ''.join(df[column_name].dropna().astype(str))

    unusual_count = df[column_name].dropna().astype(str).apply(
        lambda x: any(char not in allowed_set for char in x)
    ).sum()
    
    unusual_characters = set(all_characters) - allowed_set

    print("Number of rows containing special characters:", unusual_count)
    print("Unusual Characters Found:", unusual_characters)



class NameFeatureProcessor:
    def __init__(self,category, ngram_range = (2, 3)):
        """
        Initialize the processor with optional n-gram range for text vectorization.
        """
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.category = category

    @staticmethod
    def analyze_name(name):
        if not isinstance(name, str) or not name.strip():  # Handle empty or invalid names
            return pd.Series({
                'Length': 0,  
                'Vowel Count': 0,
                'Consonant Count': 0,
                'Vowel/Consonant Ratio': 0,
            })

        vowels = set('aeiouyüéèäöÃëçÖïá')
        consonants = set('bcdfghjklmnpqrstvwxzç')
        length = len(name)
        vowel_count = sum(1 for char in name.lower() if char in vowels)
        consonant_count = sum(1 for char in name.lower() if char in consonants)
        return pd.Series({
            'Length': length,
            'Vowel Count': vowel_count,
            'Consonant Count': consonant_count,
            'Vowel/Consonant Ratio': vowel_count / consonant_count if consonant_count > 0 else 0,
        })

    @staticmethod
    def first_last_letter(name,alphabet=None):
        """
        Create columns for the first and last letter of the name for an extended alphabet.
        Each column corresponds to a letter of the alphabet plus additional diacritic letters.
    """
        # Define the extended alphabet
        if alphabet == None:
            alphabet = 'abcdefghijklmnopqrstuvwxyzüéèäöÃëçÖïáéäÔþçÁøõãæšáàÂùðìôêÖØÀûßýÉïåÓúśíłÅÞūžâÍÈëōîñüèóöÕò'

        # Initialize all columns to 0
        columns = {f"{letter}_f": 0 for letter in alphabet}
        columns.update({f"{letter}_l": 0 for letter in alphabet})

        # Validate the input name
        if not isinstance(name, str) or not name.strip():
            return pd.Series(columns)
    
        # Get the first and last letter
        name = name.strip().lower()
        first_letter = name[0] if name else None
        last_letter = name[-1] if name else None

        # Set 1 for the corresponding first and last letter columns
        if first_letter in alphabet:
            columns[f"{first_letter}_f"] = 1
        if last_letter in alphabet:
            columns[f"{last_letter}_l"] = 1

        return pd.Series(columns)
        

    @staticmethod
    def add_diacritic_columns(names, diacritics="üéèäöÃëçÖïáéäÔþçÁøõãæšáàÂùðìôêÖØÀûßýÉïåÓúśíłÅÞūžâÍÈëōîñüèóöÕò"):
        """
        Add binary columns for each diacritic in the names.
        """
        diacritic_set = set(diacritics)
        diacritic_columns = {
            f"{diacritic}": names.apply(lambda name: 1 if diacritic in name.lower() else 0)
            for diacritic in diacritic_set
        }
        diacritic_df = pd.DataFrame(diacritic_columns)
        # Drop columns where no diacritics are found
        diacritic_df = diacritic_df.loc[:, (diacritic_df.sum(axis=0) > 0)]
        return diacritic_df

    @staticmethod
    def add_soundex_encoding(names):
        """
        Add Soundex encoding to the names.
        """
        soundex_series = names.apply(soundex)
        return pd.get_dummies(soundex_series, prefix='Soundex')

    def add_ngram_features(self, names):
        """
        Add n-gram features for the names using character-based n-grams.
        """
        if self.ngram_range is not None:
            self.vectorizer = CountVectorizer(analyzer='char', ngram_range=self.ngram_range)
            ngram_features = self.vectorizer.fit_transform(names)
            return pd.DataFrame(ngram_features.toarray(), columns=self.vectorizer.get_feature_names_out())
        return pd.DataFrame()

    def process(self, df,alphabet = None,analyze_name = True, diacritic = True, phonetics = True, first_last = True, ngram=False):
        """
        Process the input DataFrame to add all the features.
        """
        # Analyze names
        if analyze_name:
            df = df.join(df[self.category].apply(self.analyze_name))

        # Add diacritic columns
        if diacritic:
            diacritic_df = self.add_diacritic_columns(df[self.category])
            df = df.join(diacritic_df)

        # Add Soundex encoding
        if phonetics:
            soundex_df = self.add_soundex_encoding(df[self.category])
            df = pd.concat([df, soundex_df], axis=1)

        # Add first and last letter columns for the extended alphabet
        if first_last:
            letter_df = df[self.category].apply(lambda x : self.first_last_letter(x,alphabet= alphabet))
            df = pd.concat([df, letter_df], axis=1)

        # Add n-gram features
        if ngram:
            ngram_df = self.add_ngram_features(df[self.category])
            df = pd.concat([df, ngram_df], axis=1)

        return df