"""
Script containing functions used in results.
"""
import numpy as np
import pandas as pd
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import VarianceThreshold

import plotly.graph_objects as go
import pycountry
import pycountry_convert as pc
import seaborn as sns


############## Data presentation ####################

def calculate_column_freq(df, column_name='Character_name'):
    """
    Calculate the count and frequency (percentage) of unique values in a specified column.
    """
    total_entries = df[column_name].count()
    
    counts_df = df[column_name].value_counts().reset_index()
    counts_df.columns = [column_name, 'Count']
    
    counts_df['Frequency (%)'] = counts_df['Count'] / total_entries * 100
    
    return counts_df

############## Statistics ####################

def create_contingency_table(df_char_cleaned:pd.DataFrame,feature1:str,feature2:str):
    contingency_table = pd.crosstab(df_char_cleaned[feature1], df_char_cleaned[feature2])
    return contingency_table

##############  Genre Analysis ####################

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

def get_top_names_by_genre(phonetic_df, nb_of_names, genres = genres_list):

    df_male = phonetic_df[phonetic_df['Sex'] == 'M']
    df_female = phonetic_df[phonetic_df['Sex'] == 'F']

    top_male_names_by_genre = {}
    top_female_names_by_genre = {}

    # We loop through each genre to get top names for males and females
    for genre in genres:
        male_genre_names = df_male[df_male['Genre_Category'].apply(lambda categories: genre in categories)]

        top_male_names = male_genre_names['Character_name'].value_counts().head(nb_of_names).index.tolist()
        top_male_names_by_genre[genre] = top_male_names

        female_genre_names = df_female[df_female['Genre_Category'].apply(lambda categories: genre in categories)]

        top_female_names = female_genre_names['Character_name'].value_counts().head(nb_of_names).index.tolist()
        top_female_names_by_genre[genre] = top_female_names
    
    frequent_names_m = pd.DataFrame.from_dict(top_male_names_by_genre, orient='index').transpose()
    frequent_names_f = pd.DataFrame.from_dict(top_female_names_by_genre, orient='index').transpose()

    return frequent_names_m, frequent_names_f

def create_sunburst_data(frequent_names_f):
    sunburst_data = []
    sunburst_data.append({
        'character': "Movies' Genres",
        'parent': '',
    })

    for genre in frequent_names_f.columns:
        sunburst_data.append({
            'character': genre,
            'parent': "Movies' Genres", 
        })

    for genre in frequent_names_f.columns:
        for idx, prenom in enumerate(frequent_names_f[genre]):
            sunburst_data.append({
                'character': prenom,
                'parent': genre,  # Genre comme parent
            })

    sunburst_df = pd.DataFrame(sunburst_data)
    data = {
        'character': sunburst_df['character'].tolist(),
        'parent': sunburst_df['parent'].tolist(),
    }

    return data


def count_name_appearance_by_genre(df, genres=genres_list, name_substring='Luca'):
    df_name = df[df['Character_name'].str.lower().str.startswith(name_substring.lower(), na=False)]

    df_exploded = df_name.explode('Genre_Category')

    df_exploded = df_exploded[df_exploded['Genre_Category'].isin(genres)]

    genre_counts = (
        df_exploded.groupby('Genre_Category')['Wikipedia_ID']
        .nunique()
        .reindex(genres, fill_value=0)
    )

    total_count = df_exploded['Wikipedia_ID'].nunique()
 
    genre_counts_df = genre_counts.reset_index().rename(columns={'Wikipedia_ID': 'Count'})
    genre_counts_df = pd.concat([genre_counts_df, pd.DataFrame({'Genre_Category': ['Total different movies'], 'Count': [total_count]})], ignore_index=True)

    genre_counts_transposed = genre_counts_df.set_index('Genre_Category').T.reset_index(drop=True)
    genre_counts_transposed.columns.name = None

    return genre_counts_df, df_name

### ---------- Gender Analysis ---------------------

def get_vowel_stats(df_char_cleaned:pd.DataFrame):
    vowels = set('aeiouy')

    def count_vowels(name):
        return sum(1 for char in name.lower() if char in vowels)

    def count_consonants(name):
        return sum(1 for char in name.lower() if char not in vowels)

    df_char_cleaned['vowel_count'] = df_char_cleaned['Character_name'].apply(count_vowels)
    df_char_cleaned['consonant_count'] = df_char_cleaned['Character_name'].apply(count_consonants)

    stats_gender_vowels = df_char_cleaned.groupby('Sex')['vowel_count'].agg(['mean', 'std'])
    stats_gender_consonants = df_char_cleaned.groupby('Sex')['consonant_count'].agg(['mean', 'std'])
    return stats_gender_vowels, stats_gender_consonants

def get_length_stats(df_char_cleaned:pd.DataFrame):
    df_char_cleaned['name_length'] = df_char_cleaned['Character_name'].apply(lambda name: len(name))
    stats_length = df_char_cleaned.groupby('Sex')['name_length'].agg(['mean', 'std'])

    return stats_length

def create_boxenplot_by_sex(data: pd.DataFrame):
    data['name_length'] = data['Character_name'].apply(lambda name: len(name))

    fig = sns.boxenplot(data, x='Sex', y='name_length')
    fig.set_xlabel('Gender')
    fig.set_ylabel('Letter count')
    fig.set_title('Letter Count per Gender')

def get_vowel_percentage(df_char_cleaned:pd.DataFrame):
    df_char_cleaned['vowel_percentage'] = df_char_cleaned['vowel_count'] / df_char_cleaned['name_length']
    percent_vowels = df_char_cleaned.groupby('Sex')['vowel_percentage'].agg(['mean', 'std'])

    return percent_vowels

def create_letter_count_df(df, letter_position):

    df_letter = df.copy()
    df_letter['letter'] = df_letter['Character_name'].apply(lambda name: name[letter_position].lower())
    
    letter_counts_H = df_letter[df_letter['Sex'] == 'M']['letter'].value_counts()
    letter_counts_F = df_letter[df_letter['Sex'] == 'F']['letter'].value_counts()

    male_count = df_letter[df_letter['Sex'] == 'M'].shape[0]
    female_count = df_letter[df_letter['Sex'] == 'F'].shape[0]
    letter_counts_H_percentage = letter_counts_H / male_count*100
    letter_counts_F_percentage = letter_counts_F / female_count*100
    letter_counts = pd.concat([letter_counts_H_percentage, letter_counts_F_percentage], axis=1)
    letter_counts.columns = ['letter_men', 'letter_women']
    letter_counts = letter_counts.head(26)  # Limit to top 26 letters

    top_letter_names = (
        df_letter.groupby(['letter', 'Sex'])['Character_name']
        .apply(lambda x: x.value_counts().head(3).index.tolist())
        .unstack(fill_value=[])
    )

    return letter_counts, top_letter_names

def plot_letter_name_percentage(df, letter_position):

    letter_counts, top_letter_names = create_letter_count_df(df, letter_position)

    if letter_position == 0:
        title = 'Percentage of Names Starting by Each Letter by Gender'
    else:
        title = 'Percentage of Names Ending by Each Letter by Gender'

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=letter_counts.index,
        y=letter_counts['letter_men'],
        name='Male',
        marker_color='skyblue',
        hovertext=[f"Top names: {', '.join(top_letter_names.loc[letter, 'M'])}" if 'M' in top_letter_names.columns else "" for letter in letter_counts.index],
        hoverinfo="text"
    ))

    fig.add_trace(go.Bar(
        x=letter_counts.index,
        y=letter_counts['letter_women'],
        name='Female',
        marker_color='salmon',
        hovertext=[f"Top names: {', '.join(top_letter_names.loc[letter, 'F'])}" if 'F' in top_letter_names.columns else "" for letter in letter_counts.index],
        hoverinfo="text"
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Letter of the Name',
        yaxis_title='% of Total Names by Gender',
        barmode='group',
        yaxis=dict(ticksuffix='%'),
        legend_title="Gender"
    )

    fig.show()

def plot_age_sex_distribution_with_top_names(df_char_cleaned: pd.DataFrame):
    age_bins = [0, 12, 17, 24, 34, 44, 54, 64, 74, 84, 100]
    age_labels = [
        '<12y', '13y-17y', '18y-24y', '25y-34y', '35y-44y', 
        '45y-54y', '55y-64y', '65y-74y', '75y-84y', '>85y'
    ]

    df_char_cleaned['age_category'] = pd.cut(df_char_cleaned['Actor_age'], bins=age_bins, labels=age_labels, right=False)

    age_sex_counts = df_char_cleaned.groupby(['age_category', 'Sex'], observed= True).size().unstack(fill_value=0)
    total_counts = df_char_cleaned['Sex'].value_counts()
    age_sex_percentage = age_sex_counts.div(total_counts, axis=1) * 100

    top_names = (
        df_char_cleaned.groupby(['age_category', 'Sex'], observed= True)['Character_name']
        .apply(lambda x: x.value_counts().head(3).index.tolist())
        .unstack(fill_value=[])
    )
    fig = go.Figure()

    for sex in ['M', 'F']:
        fig.add_trace(go.Bar(
            x=age_labels,
            y=age_sex_percentage[sex],
            name='Male' if sex == 'M' else 'Female',
            marker_color='skyblue' if sex == 'M' else 'salmon',
            hovertext=[f"Top names: {', '.join(top_names.loc[age, sex])}" for age in age_labels],
            hoverinfo="text"
        ))
    fig.update_layout(
        title='Percentage of Males and Females in Each Age Category',
        xaxis_title='Age Category',
        yaxis_title='% of Total Males/Females',
        barmode='group',
        xaxis=dict(tickvals=age_labels, tickangle=0),
        yaxis=dict(ticksuffix='%'),
        legend=dict(title="Gender")
    )

    fig.show()

### ---------- Country Analysis ---------------------

def country_to_continent(country_name:str, countries_code:list[str]):
    try:
        # Get the alpha-2 and alpha-3 country code
        country_code_alpha2 = pycountry.countries.lookup(country_name).alpha_2
        country_code_alpha3 = pycountry.countries.lookup(country_name).alpha_3
        if country_code_alpha3 not in countries_code:
            countries_code.append(country_code_alpha3)
        continent_code = pc.country_alpha2_to_continent_code(country_code_alpha2)

        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except (KeyError, AttributeError, LookupError):
        return None 



def create_continent_df(df_char_cleaned:pd.DataFrame,countries_code:list[str])->pd.DataFrame:

    df_char_cleaned['primary_country'] = df_char_cleaned['Country'].str[0]
    df_char_cleaned['Continent'] = df_char_cleaned['primary_country'].apply(lambda x: country_to_continent(x, countries_code=countries_code))

    continents = df_char_cleaned.groupby(['Continent','Sex'])['Character_name'].agg(pd.Series.mode)
    df_continents = continents.to_frame().reset_index()
    df_continents.columns = ['Continent', 'Sex', 'Name']
    df_continents = df_continents.pivot(index='Continent',columns='Sex',values='Name').reset_index()
    df_continents.columns = ['Continent', 'Female_name', 'Male_name']
    
    # for Africa we will pick one of the names to display
    df_continents.loc[0,'Female_name'] = 'Amina*'
    df_continents.loc[0,'Male_name']='Omar*'

    return df_continents

def create_top_names_df(df_char_cleaned:pd.DataFrame)->pd.DataFrame:

    def tie_case(name_row):
        if isinstance(name_row,np.ndarray):
            name_row = name_row[0]
        return name_row

    country_top_name = df_char_cleaned.groupby(['primary_country','Sex'])['Character_name'].agg(pd.Series.mode)
    df_top_name = country_top_name.to_frame().reset_index()
    df_top_name.columns = ['primary_country', 'Sex', 'Name']
    df_top_name = df_top_name.pivot(index='primary_country',columns='Sex',values='Name').reset_index()
    df_top_name.columns = ['primary_country', 'Female_name', 'Male_name']

    df_top_name['Female_name'] = df_top_name['Female_name'].apply(tie_case) #In case of a tie we choose the 1st element
    df_top_name['Male_name'] = df_top_name['Male_name'].apply(tie_case) #In case of a tie we choose the 1st element

    return df_top_name

def add_movie_count(df_char_cleaned:pd.DataFrame, df_top_names:pd.DataFrame)->None:
    unique_name_counts = df_char_cleaned.groupby('primary_country')['Name'].nunique()
    df_top_names['Number_of_movies'] = df_top_names['primary_country'].map(unique_name_counts)

def cleaning_non_countries(df_top_names:pd.DataFrame)->pd.DataFrame:

    #For the United Kingdom
    df_top_names = df_top_names[df_top_names['primary_country'] != 'England']
    df_top_names = df_top_names[df_top_names['primary_country'] != 'Wales']
    df_top_names = df_top_names[df_top_names['primary_country'] != 'Northern Ireland']
    df_top_names = df_top_names[df_top_names['primary_country'] != 'Kingdom of Great Britain']

    #For Germany
    df_top_names = df_top_names[df_top_names['primary_country'] != 'Weimar Republic']
    df_top_names = df_top_names[df_top_names['primary_country'] != 'West Germany']
    df_top_names = df_top_names[df_top_names['primary_country'] != 'German Democratic Republic']

    #For Russia
    df_top_names['primary_country'] = df_top_names['primary_country'].replace('Soviet Union','Russia')

    df_top_names = (
    df_top_names.groupby('primary_country', as_index=False).agg({'Number_of_movies': 'sum', 'Female_name': 'first','Male_name': 'first'}))
    return df_top_names




### ---------- NGram Analysis ---------------------

'''def top_genre_search(genres:list[str]):
    for genre in genres:
        if genre in top_genres:
            return genre         
    return 'other'

### ---------- N-gram Analysis ---------------------

def create_ngram(df_char_cleaned:pd.DataFrame):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
    char_ngrams = vectorizer.fit_transform(df_char_cleaned['Character_name'])
    ngram_df = pd.DataFrame(char_ngrams.toarray(), columns=vectorizer.get_feature_names_out())
    ngram_df = ngram_df.astype('float32') # converting to float32 to decrease the computing time

    return ngram_df

def kmeans_clustering(ngram_df:pd.DataFrame,df_char_cleaned:pd.DataFrame):
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=1000, random_state=42) 
    df_char_cleaned['cluster'] = kmeans.fit_predict(ngram_df)

def ipca_reduction(ngram_df:pd.DataFrame):
    ipca = IncrementalPCA(n_components=3, batch_size=500)
    pca_result = ipca.fit_transform(ngram_df)
    loadings = pd.DataFrame(ipca.components_.T, columns=[f'PC{i+1}' for i in range(ipca.n_components_)], index=ngram_df.columns)
    loadings['PC1']=loadings['PC1'].apply(abs)

    return pca_result,loadings

def create_df_country(df_char_cleaned:pd.DataFrame,pca_result:pd.DataFrame):
    df_country = df_char_cleaned.copy()
    df_country['pca_one'] = pca_result[:, 0]
    df_country['pca_two'] = pca_result[:, 1]
    df_country['pca_three'] = pca_result[:, 2]
    return df_country
    '''