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
    names=['Wikipedia_ID', 'Freebase_ID', 'Release_date1', 'Character_name', 
           'Actor_DOB', 'Sex', 'Height', 'Ethnicity', 'Actor_name', 
           'Actor_age', 'Freebase_character_map', '?', '??']
)

# Remove rows with NaN in 'Character_name' and filter the english common names
character_df = remove_nan_rows(character_df, 'Character_name')
character_df['Character_name'] = character_df['Character_name'].apply(filter_non_english_names)

# Find the index of the words removed aboved
df_character_filtered = character_df.copy()
df_character_filtered['Character_name']=df_character_filtered['Character_name'].apply(filter_non_english_names)

# Find the words removed
values_filtered = df_character_filtered['Character_name'].value_counts()
deleted_names = character_df[df_character_filtered['Character_name']==values_filtered.index[0]]

# Create a dataframe that contains the names filtered
kept_names = df_character_filtered[df_character_filtered['Character_name']!=values_filtered.index[0]]
print('Number of names kept:', kept_names.shape[0])

# Looking into the words removed if there is some names from a dataset
deleted_names_saved = deleted_names.copy()
deleted_names_saved['Character_name']=deleted_names_saved['Character_name'].apply(keep_names)

values_saved = deleted_names_saved['Character_name'].value_counts()
print('Number of names saved:', deleted_names_saved.shape[0])
print(values_saved)
saved_names = deleted_names_saved[deleted_names_saved['Character_name']!=values_saved.index[0]]
print('Number of names saved back:', saved_names.shape[0])

# Concatenate the filtered names and the saved names
kept_names = pd.concat([kept_names, saved_names])
kept_names['Character_name'] = kept_names['Character_name'].apply(keep_first_name)

# Merged the filtered character names with the movies dataset to add some informations on the dataset
df_char_cleaned = pd.merge(movies_df,kept_names, on="Wikipedia_ID",how="inner")[['Wikipedia_ID','Name','Languages','Country','Genres','Character_name','Sex','Actor_age','Release_date']]
print('Number of rows in the cleaned dataframe:', df_char_cleaned.shape[0])

df_char_cleaned.to_csv('data/cleaned.csv', index=False)