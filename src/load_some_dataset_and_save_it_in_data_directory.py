import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from data_utils import str_dict_to_values, remove_nan_rows, filter_non_english_names, keep_names, keep_first_name

# Load movie and character metadata from TSV files into DataFrames
movie_df = pd.read_csv(
    'MovieSummaries/movie.metadata.tsv', sep='\t', 
    names=['Wikipedia_ID', 'Freebase_ID', 'Name', 'Release_date', 
           'Box_office_revenues', 'Runtime', 'Languages', 'Country', 'Genres']
)
character_df = pd.read_csv(
    'MovieSummaries/character.metadata.tsv', sep='\t', 
    names=['Wikipedia_ID', 'Freebase_ID', 'Release_date', 'Character_name', 
           'Actor_DOB', 'Sex', 'Height', 'Ethnicity', 'Actor_name', 
           'Actor_age', 'Freebase_character_map', '?', '??']
)

# Clean 'Languages', 'Country', and 'Genres' columns in movie_df
movie_df['Languages'] = movie_df['Languages'].apply(str_dict_to_values)
movie_df['Country'] = movie_df['Country'].apply(str_dict_to_values)
movie_df['Genres'] = movie_df['Genres'].apply(str_dict_to_values)

# Remove rows with NaN values in the 'Character_name' column in character_df
character_df = remove_nan_rows(character_df, 'Character_name')

# Filter non-English character names
df_character_filtered = character_df.copy()
df_character_filtered['Character_name'] = df_character_filtered['Character_name'].apply(filter_non_english_names)

# Process common names
values_filtered = character_df['Character_name'].value_counts()
deleted_names = character_df[character_df['Character_name'] == values_filtered.index[0]]

# Keep only selected names after filtering
kept_names = df_character_filtered[df_character_filtered['Character_name'] != values_filtered.index[0]]
deleted_names_filtered = deleted_names.copy()
deleted_names_filtered['Character_name'] = deleted_names_filtered['Character_name'].apply(keep_names)
saved_names = deleted_names_filtered[deleted_names_filtered['Character_name'] != '']

# Concatenate kept and saved names, keeping only first names
kept_names = pd.concat([kept_names, saved_names], ignore_index=True)
kept_names['Character_name'] = kept_names['Character_name'].apply(keep_first_name)

# Merge movies_df with kept_names and retain relevant columns
df_cleaned = pd.merge(
    movie_df, kept_names, on="Wikipedia_ID", how="inner"
)[['Wikipedia_ID', 'Name', 'Languages', 'Country', 'Genres', 'Character_name', 'Sex']]
df_cleaned = remove_nan_rows(df_cleaned, 'Character_name')

# Save cleaned DataFrame to CSV
df_cleaned.to_csv('df_cleaned.csv', index=False)
