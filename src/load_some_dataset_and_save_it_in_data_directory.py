import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import plotly.express as px
import geopandas as gpd
from data_utils import *

from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import VarianceThreshold
from metaphone import doublemetaphone

# Load movie and character metadata from a TSV file into DataFrames
movies_df = pd.read_csv('MovieSummaries/movie.metadata.tsv',sep='\t',names=['Wikipedia_ID','Freebase_ID','Name','Release_date','Box_office_revenues','Runtime','Languages','Country','Genres'])
character_df = pd.read_csv('MovieSummaries/character.metadata.tsv',sep='\t', names=['Wikipedia_ID','Freebase_ID','Release_date','Character_name','Actor_DOB','Sex','Height','Ethnicity','Actor_name','Actor_age','Freebase_character_map','?','??'])

# Cleaning Languages, Country and Genres Columns
movies_df['Languages'] = movies_df['Languages'].apply(str_dict_to_values)
movies_df['Country'] = movies_df['Country'].apply(str_dict_to_values)
movies_df['Genres'] = movies_df['Genres'].apply(str_dict_to_values)

# Remove NaN rows for Character Names
character_df = remove_nan_rows(character_df,'Character_name')

# Common names cleaning
df_character_filtered = character_df.copy()
df_character_filtered['Character_name']=df_character_filtered['Character_name'].apply(filter_non_english_names)

values_filtered = character_df['Character_name'].value_counts()
deleted_names = character_df[character_df['Character_name']==values_filtered.index[0]]

kept_names = df_character_filtered[df_character_filtered['Character_name']!=values_filtered.index[0]]

deleted_names_filtered = deleted_names.copy()
deleted_names_filtered['Character_name']=deleted_names_filtered['Character_name'].apply(keep_names)

saved_names = deleted_names_filtered[deleted_names_filtered['Character_name']!='']

kept_names = pd.concat([kept_names, saved_names], ignore_index=True)

kept_names['Character_name'] = kept_names['Character_name'].apply(keep_first_name)

# Merging with movie.metadata.tsv
df_cleaned = pd.merge(movies_df,kept_names, on="Wikipedia_ID",how="inner")[['Wikipedia_ID','Name','Languages','Country','Genres','Character_name','Sex']]
df_cleaned = remove_nan_rows(df_cleaned,'Character_name')

df_cleaned.to_csv('df_cleaned.csv', index=False)