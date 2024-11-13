"""
Script for leading and preprocessing the data needed.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from utils.data_utils import *

# Load movie and character metadata from TSV files
movies_df = pd.read_csv(
    'src/MovieSummaries/movie.metadata.tsv', sep='\t', 
    names=['Wikipedia_ID', 'Freebase_ID', 'Name', 'Release_date', 
           'Box_office_revenues', 'Runtime', 'Languages', 'Country', 'Genres']
)
character_df = pd.read_csv(
    'src/MovieSummaries/character.metadata.tsv', sep='\t', 
    names=['Wikipedia_ID', 'Freebase_ID', 'Release_date', 'Character_name', 
           'Actor_DOB', 'Sex', 'Height', 'Ethnicity', 'Actor_name', 
           'Actor_age', 'Freebase_character_map', '?', '??']
)

# Remove rows with NaN in 'Character_name' and filter non-English names
character_df = remove_nan_rows(character_df, 'Character_name')
character_df['Character_name'] = character_df['Character_name'].apply(filter_non_english_names)

# Process common names (j'ai rien comprs à cette partie)
most_common_name = character_df['Character_name'].value_counts().idxmax()
kept_names = character_df[character_df['Character_name'] != most_common_name]

# Handle deleted names and keep only meaningful names
deleted_names = character_df[character_df['Character_name'] == most_common_name].copy()
deleted_names['Character_name'] = deleted_names['Character_name'].apply(keep_names)
saved_names = deleted_names[deleted_names['Character_name'] != '']

# Concatenate filtered names and keep only first names
kept_names = pd.concat([kept_names, saved_names], ignore_index=True)
kept_names['Character_name'] = kept_names['Character_name'].apply(keep_first_name)

# Merge cleaned DataFrames and save the result
df_cleaned = pd.merge(
    movies_df, kept_names, on="Wikipedia_ID", how="inner"
)[['Wikipedia_ID', 'Name', 'Languages', 'Country', 'Genres', 'Character_name', 'Sex']]
df_cleaned = remove_nan_rows(df_cleaned, 'Character_name')

df_cleaned.to_csv('data/cleaned.csv', index=False)
