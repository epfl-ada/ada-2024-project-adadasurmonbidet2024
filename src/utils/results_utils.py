"""
Script containing functions used in results.
"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import VarianceThreshold

import pycountry
import pycountry_convert as pc

genres_list = ['Action & Adventure', 'Drama', 'Comedy', 'Horror & Thriller', 
              'Fantasy & Sci-Fi', 'Historical & War', 'Romance', 'Documentary', 
              'Music & Performance', 'Cult & B-Movies', 'Other']

def categorize_genre(genre_list: list) -> list:
    action_adventure = ['Action', 'Adventure', 'Thriller', 'War film', 'Action/Adventure', 'Martial Arts Film', 'Wuxia', 'Superhero movie', 'Western', 'Sword and sorcery', 'Spy', 'Supernatural']
    drama = ['Drama', 'Biographical film', 'Crime Drama', 'Family Film', 'Family Drama', 'Historical fiction', 'Biopic [feature]', 'Courtroom Drama', 'Political drama', 'Family-Oriented Adventure', 'Psychological thriller']
    comedy = ['Comedy', 'Romantic comedy', 'Comedy-drama', 'Comedy film', 'Black comedy', 'Slapstick', 'Romantic comedy', 'Musical', 'Satire', 'Parody', 'Comedy horror']
    horror_thriller = ['Horror', 'Psychological horror', 'Horror Comedy', 'Slasher', 'Thriller', 'Crime Thriller', 'Sci-Fi Horror', 'Suspense', 'Zombie Film', 'Natural horror films']
    fantasy_sci = ['Fantasy', 'Science Fiction', 'Space western', 'Fantasy Adventure', 'Fantasy Comedy', 'Sci-Fi Horror', 'Sci-Fi Thriller', 'Fantasy Drama', 'Dystopia', 'Alien Film', 'Cyberpunk', 'Time travel']
    historical_war = ['Historical drama', 'Historical fiction', 'Historical Epic', 'Epic', 'War effort', 'War film', 'Period piece', 'Courtroom Drama']
    romance = ['Romance Film', 'Romantic drama', 'Romance', 'Romantic fantasy', 'Marriage Drama']
    documentary = ['Documentary', 'Docudrama', 'Biography', 'Historical Documentaries', 'Mondo film', 'Patriotic film', 'Educational']
    music_performance = ['Musical', 'Music', 'Musical Drama', 'Musical comedy', 'Dance', 'Jukebox musical', 'Concert film']
    cult_b_movies = ['Cult', 'B-movie', 'Indie', 'Experimental film', 'Surrealism', 'Avant-garde', 'Grindhouse', 'Blaxploitation', 'Camp']

    categories = []

    for genre in genre_list:
        if genre in action_adventure:
            if 'Action & Adventure' not in categories:
                categories.append('Action & Adventure')
        if genre in drama:
            if 'Drama' not in categories:
                categories.append('Drama')
        if genre in comedy:
            if 'Comedy' not in categories:
                categories.append('Comedy')
        if genre in horror_thriller:
            if 'Horror & Thriller' not in categories:
                categories.append('Horror & Thriller')
        if genre in fantasy_sci:
            if 'Fantasy & Sci-Fi' not in categories:
                categories.append('Fantasy & Sci-Fi')
        if genre in historical_war:
            if 'Historical & War' not in categories:
                categories.append('Historical & War')
        if genre in romance:
            if 'Romance' not in categories:
                categories.append('Romance')
        if genre in documentary:
            if 'Documentary' not in categories:
                categories.append('Documentary')
        if genre in music_performance:
            if 'Music & Performance' not in categories:
                categories.append('Music & Performance')
        if genre in cult_b_movies:
            if 'Cult & B-Movies' not in categories:
                categories.append('Cult & B-Movies')

    return categories if categories else ['Other']

def get_top_names_by_genre(phonetic_df, genres = genres_list):

    df_male = phonetic_df[phonetic_df['Sex'] == 'M']
    df_female = phonetic_df[phonetic_df['Sex'] == 'F']

    top_male_names_by_genre = {}
    top_female_names_by_genre = {}

    # Loop through each genre and get top names for males and females
    for genre in genres:
        male_genre_names = df_male[df_male['Genre_Category'].apply(lambda categories: genre in categories)]

        top_male_names = male_genre_names['Character_name'].value_counts().head(10).index.tolist()
        top_male_names_by_genre[genre] = top_male_names

        female_genre_names = df_female[df_female['Genre_Category'].apply(lambda categories: genre in categories)]

        top_female_names = female_genre_names['Character_name'].value_counts().head(10).index.tolist()
        top_female_names_by_genre[genre] = top_female_names
    
    # Convert dictionaries to DataFrames with each genre as a column
    frequent_names_m = pd.DataFrame.from_dict(top_male_names_by_genre, orient='index').transpose()
    frequent_names_f = pd.DataFrame.from_dict(top_female_names_by_genre, orient='index').transpose()

    return frequent_names_m, frequent_names_f

def count_name_appearance_by_genre(df, genres=genres_list, name='Tom'):
    # Filter the DataFrame for the specified name
    df_name = df[df['Character_name'] == name]

    # Initialize genre counts dictionary
    genre_counts = {genre: 0 for genre in genres}

    # Count occurrences by genre
    for _, row in df_name.iterrows():
        row_genres = row['Genre_Category']
        if isinstance(row_genres, list):
            for genre in row_genres:
                if genre in genre_counts:
                    genre_counts[genre] += 1
        else:
            if row_genres in genre_counts:
                genre_counts[row_genres] += 1

    # Convert genre counts to DataFrame
    genre_counts_df = pd.DataFrame([genre_counts])

    return genre_counts_df, df_name

### ---------- Country Analysis ---------------------

def country_to_continent(country_name:str, countries_code:list[str]):
    try:
        # Get the alpha-2 country code
        country_code_alpha2 = pycountry.countries.lookup(country_name).alpha_2
        country_code_alpha3 = pycountry.countries.lookup(country_name).alpha_3
        if country_code_alpha3 not in countries_code:
            countries_code.append(country_code_alpha3)
        continent_code = pc.country_alpha2_to_continent_code(country_code_alpha2)

        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except (KeyError, AttributeError, LookupError):
        return None 



def create_continent_df(df_char_cleaned:pd.DataFrame)->pd.DataFrame:
    df_char_cleaned['primary_country'] = df_char_cleaned['Country'].str[0]
    df_char_cleaned['Continent'] = df_char_cleaned['primary_country'].apply(country_to_continent)

    continents = df_char_cleaned.groupby(['Continent','Sex'])['Character_name'].agg(pd.Series.mode)
    df_continents = continents.to_frame().reset_index()
    df_continents.columns = ['Continent', 'Sex', 'Name']
    df_continents = df_continents.pivot(index='Continent',columns='Sex',values='Name').reset_index()
    df_continents.columns = ['Continent', 'Female_name', 'Male_name']
    
    # for Africa we will pick one of the names to display
    df_continents.loc[0,'Female_name'] = 'Amina*'
    df_continents.loc[0,'Male_name']='Omar*'

    return df_continents

