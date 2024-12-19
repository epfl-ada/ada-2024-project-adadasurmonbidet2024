"""
Script containing functions used in results.
"""
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import pycountry
import pycountry_convert as pc
import seaborn as sns
from metaphone import doublemetaphone

import gzip
import xml.etree.ElementTree as Xet
from textblob import TextBlob

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

def plot_sex_distribution_with_top_names(df_char_cleaned: pd.DataFrame):
    # Calculate total counts for each gender
    sex_counts = df_char_cleaned.groupby('Sex').size()
    
    # Get the top 3 frequent names for each gender
    top_names = (
        df_char_cleaned.groupby('Sex')['Character_name']
        .apply(lambda x: x.value_counts().head(3).index.tolist())
    )

    # Create the plot
    fig = go.Figure()

    # Add bars for male and female
    for sex, color in zip(['M', 'F'], ['skyblue', 'salmon']):
        fig.add_trace(go.Bar(
            x=['Male' if sex == 'M' else 'Female'],  # Male or Female as x-axis categories
            y=[sex_counts[sex]],  # Total counts for each gender
            name='Male' if sex == 'M' else 'Female',
            marker_color=color,
            hovertext=[
                f"Count: {sex_counts[sex]}<br>Top Names: {', '.join(top_names[sex])}"
            ],
            hoverinfo="text"
        ))

    # Update layout
    fig.update_layout(
        title='Total Names Count by Gender with Top 3 Names',
        xaxis_title='Gender',
        yaxis_title='Name Count',
        xaxis=dict(tickangle=0),
    )

    # Show the plot
    fig.show()

############## Statistics ####################

def create_contingency_table(df_char_cleaned:pd.DataFrame,feature1:str,feature2:str):
    contingency_table = pd.crosstab(df_char_cleaned[feature1], df_char_cleaned[feature2])
    return contingency_table

##############  Genre Analysis ####################
class Analyzer:
    def __init__(self, data):
        self.data = data


class GenreAnalyzer(Analyzer):
    def __init__(self, data):
        super().__init__(data)
        self.genres_list = ['Action & Adventure', 'Drama', 'Comedy', 'Horror & Thriller', 
              'Fantasy & Sci-Fi', 'Historical & War', 'Romance', 'Documentary', 
              'Music & Performance', 'Cult & B-Movies', 'Other']
        #self.data["Genre_Category"] = self.data['Genres'].apply(lambda x: self._categorize_genre(x))
    

    # def _categorize_genre(self,genres_movies) -> list:
    #     action_adventure = ['Action', 'Adventure', 'Thriller', 'War film', 'Action/Adventure', 'Martial Arts Film', 'Wuxia', 'Superhero movie', 'Western', 'Sword and sorcery', 'Spy', 'Supernatural']
    #     drama = ['Drama', 'Biographical film', 'Crime Drama', 'Family Film', 'Family Drama', 'Historical fiction', 'Biopic [feature]', 'Courtroom Drama', 'Political drama', 'Family-Oriented Adventure', 'Psychological thriller']
    #     comedy = ['Comedy', 'Romantic comedy', 'Comedy-drama', 'Comedy film', 'Black comedy', 'Slapstick', 'Romantic comedy', 'Musical', 'Satire', 'Parody', 'Comedy horror']
    #     horror_thriller = ['Horror', 'Psychological horror', 'Horror Comedy', 'Slasher', 'Thriller', 'Crime Thriller', 'Sci-Fi Horror', 'Suspense', 'Zombie Film', 'Natural horror films']
    #     fantasy_sci = ['Fantasy', 'Science Fiction', 'Space western', 'Fantasy Adventure', 'Fantasy Comedy', 'Sci-Fi Horror', 'Sci-Fi Thriller', 'Fantasy Drama', 'Dystopia', 'Alien Film', 'Cyberpunk', 'Time travel']
    #     historical_war = ['Historical drama', 'Historical fiction', 'Historical Epic', 'Epic', 'War effort', 'War film', 'Period piece', 'Courtroom Drama']
    #     romance = ['Romance Film', 'Romantic drama', 'Romance', 'Romantic fantasy', 'Marriage Drama']
    #     documentary = ['Documentary', 'Docudrama', 'Biography', 'Historical Documentaries', 'Mondo film', 'Patriotic film', 'Educational']
    #     music_performance = ['Musical', 'Music', 'Musical Drama', 'Musical comedy', 'Dance', 'Jukebox musical', 'Concert film']
    #     cult_b_movies = ['Cult', 'B-movie', 'Indie', 'Experimental film', 'Surrealism', 'Avant-garde', 'Grindhouse', 'Blaxploitation', 'Camp']

    #     categories = []

    #     for genre in genres_movies:
    #         if genre in action_adventure:
    #             if 'Action & Adventure' not in categories:
    #                 categories.append('Action & Adventure')
    #         if genre in drama:
    #             if 'Drama' not in categories:
    #                 categories.append('Drama')
    #         if genre in comedy:
    #             if 'Comedy' not in categories:
    #                 categories.append('Comedy')
    #         if genre in horror_thriller:
    #             if 'Horror & Thriller' not in categories:
    #                 categories.append('Horror & Thriller')
    #         if genre in fantasy_sci:
    #             if 'Fantasy & Sci-Fi' not in categories:
    #                 categories.append('Fantasy & Sci-Fi')
    #         if genre in historical_war:
    #             if 'Historical & War' not in categories:
    #                 categories.append('Historical & War')
    #         if genre in romance:
    #             if 'Romance' not in categories:
    #                 categories.append('Romance')
    #         if genre in documentary:
    #             if 'Documentary' not in categories:
    #                 categories.append('Documentary')
    #         if genre in music_performance:
    #             if 'Music & Performance' not in categories:
    #                 categories.append('Music & Performance')
    #         if genre in cult_b_movies:
    #             if 'Cult & B-Movies' not in categories:
    #                 categories.append('Cult & B-Movies')

    #     return categories if categories else ['Other']

class GenreAnalyzer(Analyzer):
    def __init__(self, data):
        super().__init__(data)
        self.genres_list = ['Action & Adventure', 'Drama', 'Comedy', 'Horror & Thriller', 
              'Fantasy & Sci-Fi', 'Historical & War', 'Romance', 'Documentary', 
              'Music & Performance', 'Cult & B-Movies', 'Other']
        self.data["Genre_Category"] = self.data['Genres'].apply(lambda x: categorize_genre(x))
    

    def get_top_names_by_genre(self, nb_of_names):
        genres = self.genres_list
        df_male = self.data[self.data['Sex'] == 'M']
        df_female = self.data[self.data['Sex'] == 'F']

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

    def create_sunburst_data(self,frequent_names_f):
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

    def count_name_appearance_by_genre(self, name_substring='Luca'):
        df_name = self.data[self.data['Character_name'].str.lower().str.startswith(name_substring.lower(), na=False)]

        df_exploded = df_name.explode('Genre_Category')

        df_exploded = df_exploded[df_exploded['Genre_Category'].isin(self.genres_list)]

        genre_counts = (
            df_exploded.groupby('Genre_Category')['Wikipedia_ID']
            .nunique()
            .reindex(self.genres_list, fill_value=0)
        )

        total_count = df_exploded['Wikipedia_ID'].nunique()
    
        genre_counts_df = genre_counts.reset_index().rename(columns={'Wikipedia_ID': 'Count'})
        genre_counts_df = pd.concat([genre_counts_df, pd.DataFrame({'Genre_Category': ['Total different movies'], 'Count': [total_count]})], ignore_index=True)

        genre_counts_transposed = genre_counts_df.set_index('Genre_Category').T.reset_index(drop=True)
        genre_counts_transposed.columns.name = None

        return genre_counts_df, df_name
    
    def get_top_names_by_genre_SA(self, nb_of_names):
        genres = self.genres_list
        df_male = self.data[self.data['Sex'] == 'M']
        df_female = self.data[self.data['Sex'] == 'F']

        results = {}

        for genre in genres:
            male_genre_names = df_male[df_male['Genres'].apply(lambda categories: genre in categories)]
            female_genre_names = df_female[df_female['Genres'].apply(lambda categories: genre in categories)]

            top_male_positive_names = male_genre_names.nlargest(nb_of_names, 'Polarity')['Character_Name'].tolist()
            top_female_positive_names = female_genre_names.nlargest(nb_of_names, 'Polarity')['Character_Name'].tolist()

            top_male_negative_names = male_genre_names.nsmallest(nb_of_names, 'Polarity')['Character_Name'].tolist()
            top_female_negative_names = female_genre_names.nsmallest(nb_of_names, 'Polarity')['Character_Name'].tolist()

            results[genre] = {
                'Female Positive Names': top_female_positive_names,
                'Male Positive Names': top_male_positive_names,
                'Female Negative Names': top_female_negative_names,
                'Male Negative Names': top_male_negative_names
            }

        result_df = pd.DataFrame.from_dict(results, orient='index')

        return result_df

### ---------- Gender Analysis ---------------------

class GenderAnalyzer(Analyzer):
    def __init__(self, data):
        super().__init__(data)

    def get_vowel_stats(self):
        vowels = set('aeiouy')

        def count_vowels(name):
            return sum(1 for char in name.lower() if char in vowels)

        def count_consonants(name):
            return sum(1 for char in name.lower() if char not in vowels)

        self.data['vowel_count'] = self.data['Character_name'].apply(count_vowels)
        self.data['consonant_count'] = self.data['Character_name'].apply(count_consonants)

        stats_gender_vowels = self.data.groupby('Sex')['vowel_count'].agg(['mean', 'std'])
        stats_gender_consonants = self.data.groupby('Sex')['consonant_count'].agg(['mean', 'std'])
        return stats_gender_vowels, stats_gender_consonants

    def get_length_stats(self):
        self.data['name_length'] = self.data['Character_name'].apply(lambda name: len(name))
        stats_length = self.data.groupby('Sex')['name_length'].agg(['mean', 'std'])

        return stats_length

    def create_boxenplot_by_sex(self):
        self.data['name_length'] = self.data['Character_name'].apply(lambda name: len(name))

        fig = sns.boxenplot(self.data, x='Sex', y='name_length')
        fig.set_xlabel('Gender')
        fig.set_ylabel('Name Length')
        fig.set_title('Name Length per Gender')

    def get_vowel_percentage(self):
        self.data['vowel_percentage'] = self.data['vowel_count'] / self.data['name_length']
        percent_vowels = self.data.groupby('Sex')['vowel_percentage'].agg(['mean', 'std'])

        return percent_vowels

    def _create_letter_count_df(self, letter_position):

        df_letter = self.data.copy()
        df_letter['letter'] = df_letter['Character_name'].apply(lambda name: name[letter_position].lower())
        
        letter_counts_H = df_letter[df_letter['Sex'] == 'M']['letter'].value_counts()
        letter_counts_F = df_letter[df_letter['Sex'] == 'F']['letter'].value_counts()

        male_count = df_letter[df_letter['Sex'] == 'M'].shape[0]
        female_count = df_letter[df_letter['Sex'] == 'F'].shape[0]
        letter_counts_H_percentage = letter_counts_H / male_count
        letter_counts_F_percentage = letter_counts_F / female_count
        letter_counts = pd.concat([letter_counts_H_percentage, letter_counts_F_percentage], axis=1)
        letter_counts.columns = ['letter_men', 'letter_women']
        letter_counts = letter_counts.head(26)  # Limit to top 26 letters

        top_letter_names = (
            df_letter.groupby(['letter', 'Sex'])['Character_name']
            .apply(lambda x: x.value_counts().head(3).index.tolist())
            .unstack(fill_value=[])
        )

        return letter_counts, top_letter_names

    def plot_letter_name_percentage(self, letter_position):

        letter_counts, top_letter_names = self._create_letter_count_df(letter_position)

        if letter_position == 0:
            title = 'Distribution of Names Starting by each Letter by Gender'
        else:
            title = 'Distribution of Names Ending by each Letter by Gender'

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
            xaxis_title='Letter',
            yaxis_title='Normalized Count',
            barmode='group',
            yaxis=dict(ticksuffix='%'),
        )

        fig.show()

    def plot_age_sex_distribution_with_top_names(self):
        age_bins = [0, 12, 17, 24, 34, 44, 54, 64, 74, 84, 100]
        age_labels = [
            '<12y', '13y-17y', '18y-24y', '25y-34y', '35y-44y', 
            '45y-54y', '55y-64y', '65y-74y', '75y-84y', '>85y'
        ]

        self.data['age_category'] = pd.cut(self.data['Actor_age'], bins=age_bins, labels=age_labels, right=False)

        age_sex_counts = self.data.groupby(['age_category', 'Sex'], observed= True).size().unstack(fill_value=0)
        total_counts = self.data['Sex'].value_counts()
        age_sex_percentage = age_sex_counts.div(total_counts, axis=1) * 100

        top_names = (
            self.data.groupby(['age_category', 'Sex'], observed= True)['Character_name']
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

class CountryAnalyzer(Analyzer):
    def __init__(self, data):
        super().__init__(data)
        self.countries_code = []

    def country_to_continent(self,country_name:str, countries_code:list[str]):
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

    def create_continent_df(self)->pd.DataFrame:

        self.data['primary_country'] = self.data['Country'].str[0]
        self.data['Continent'] = self.data['primary_country'].apply(lambda x: self.country_to_continent(x, self.countries_code))

        continents = self.data.groupby(['Continent','Sex'])['Character_name'].agg(pd.Series.mode)
        df_continents = continents.to_frame().reset_index()
        df_continents.columns = ['Continent', 'Sex', 'Name']
        df_continents = df_continents.pivot(index='Continent',columns='Sex',values='Name').reset_index()
        df_continents.columns = ['Continent', 'Female_name', 'Male_name']
        
        # for Africa we will pick one of the names to display
        df_continents.loc[0,'Female_name'] = 'Amina*'
        df_continents.loc[0,'Male_name']='Omar*'

        return df_continents

    def create_top_names_df(self)->pd.DataFrame:

        def tie_case(name_row):
            if isinstance(name_row,np.ndarray):
                name_row = name_row[0]
            return name_row

        country_top_name = self.data.groupby(['primary_country','Sex'])['Character_name'].agg(pd.Series.mode)
        df_top_name = country_top_name.to_frame().reset_index()
        df_top_name.columns = ['primary_country', 'Sex', 'Name']
        df_top_name = df_top_name.pivot(index='primary_country',columns='Sex',values='Name').reset_index()
        df_top_name.columns = ['primary_country', 'Female_name', 'Male_name']

        df_top_name['Female_name'] = df_top_name['Female_name'].apply(tie_case) #In case of a tie we choose the 1st element
        df_top_name['Male_name'] = df_top_name['Male_name'].apply(tie_case) #In case of a tie we choose the 1st element

        return df_top_name

    def add_movie_count(self, df_top_names:pd.DataFrame)->None:
        unique_name_counts = self.data.groupby('primary_country')['Name'].nunique()
        df_top_names['Number_of_movies'] = df_top_names['primary_country'].map(unique_name_counts)

    def cleaning_non_countries(self,df_top_names:pd.DataFrame)->pd.DataFrame:

        #For the United Kingdom
        df_top_names = df_top_names[df_top_names['primary_country'] != 'England']
        df_top_names = df_top_names[df_top_names['primary_country'] != 'Wales']
        df_top_names = df_top_names[df_top_names['primary_country'] != 'Northern Ireland']
        df_top_names = df_top_names[df_top_names['primary_country'] != 'Kingdom of Great Britain']

        #For Germany
        df_top_names = df_top_names[df_top_names['primary_country'] != 'Weimar Republic']
        df_top_names = df_top_names[df_top_names['primary_country'] != 'West Germany']
        df_top_names = df_top_names[df_top_names['primary_country'] != 'German Democratic Republic']
        df_top_names = df_top_names[df_top_names['primary_country'] != 'Nazi Germany']

        #For Russia
        df_top_names['primary_country'] = df_top_names['primary_country'].replace('Soviet Union','Russia')

        df_top_names = (
        df_top_names.groupby('primary_country', as_index=False).agg({'Number_of_movies': 'sum', 'Female_name': 'first','Male_name': 'first'}))
        return df_top_names



### ---------- Phonetic Analysis ---------------------

class PhoneticAnalyzer(Analyzer):
    def __init__(self, data,manner_groups,manner_names):
        super().__init__(data)
        self.data['Phonetic'] = self.data["Character_name"].apply(lambda x: doublemetaphone(x)[0])
        self.manner_groups = manner_groups
        self.manner_names = manner_names


    def assign_phonetic_group(self,category):
        results = []

        for i, consonants in enumerate(self.manner_groups):
            for _, row in self.data.iterrows():
                if any(consonant in row['Phonetic'] for consonant in consonants):
                    results.append({
                        f'{category}': row[f'{category}'],
                        'Consonant_Group': self.manner_names[i]
                    })
        results_df = pd.DataFrame(results)
        return results_df


    def consonant_group_features(self):
        for i, consonants in enumerate(self.manner_groups):
            group_name = self.manner_names[i]
            self.data[group_name] = self.data['Phonetic'].apply(
                lambda phonetic: 1 if any(consonant in phonetic for consonant in consonants) else 0
            )

    '''
    def compute_binomial_ci(self,p,tot_sample):
        Z = 1.96
        if tot_sample <=0:
            raise ValueError('not good tot_sample')
        if p<0:
            raise ValueError('not good p') 
        ci = Z * np.sqrt(p*(1-p)/tot_sample)
        return ci
    '''

    def phonetics_by_gender(self)->pd.DataFrame:
        tot_nb_names = self.data['Sex'].value_counts()
        manner_df = self.assign_phonetic_group('Sex')
        manner_df = manner_df.groupby(['Consonant_Group','Sex'])['Sex'].size().reset_index(name='Count')

        #We divide by the total number of female/male name to normalize the values
        manner_df['Percent']=manner_df.apply(lambda row: row['Count'] / tot_nb_names[0] if row['Sex'] == 'M' else row['Count'] / tot_nb_names[1],axis=1)

        #We create a new column calculating the CI in preparation for the plot
        #manner_df['CI'] = manner_df.apply(lambda row: self.compute_binomial_ci(row['Percent'],
                        #(tot_nb_names[0] if row['Sex'] == 'M' else tot_nb_names[1])),axis=1)
        manner_df['Percent'] *= 100
        return manner_df
    
    def phonetics_by_age(self)->pd.DataFrame:
        age_order = ['<12y','13y-17y','18y-24y','25y-34y','35y-44y','45y-54y','55y-64y','65y-74y','75y-84y','>85y']
        tot_nb_names_per_age = self.data['age_category'].value_counts().reset_index()


        manner_age_df = self.assign_phonetic_group('age_category')
        manner_age_df = manner_age_df.groupby(['Consonant_Group','age_category'])['age_category'].size().reset_index(name='Count')

        #We divide by the total number of characters in each age category to normalize the values
        manner_age_df['Percent']=manner_age_df.apply(lambda row: row['Count'] /
            tot_nb_names_per_age[tot_nb_names_per_age['age_category']==row['age_category']]['count'].values[0],axis=1) *100
        return manner_age_df




### ---------- Sentimental Analysis ---------------------

def parse_xml_gz(xml_gz_file):
    with gzip.open(xml_gz_file, 'rt', encoding='utf-8') as f:

        tree = Xet.parse(f)
        root = tree.getroot()

        sentences_data = []
        coreferences_data = []
        
        for sentence in root.findall(".//sentence"):
            sentence_text = " ".join(token.find("word").text for token in sentence.findall(".//token"))
            sentence_data = {
                "sentence_id": sentence.get("id"),
                "sentence_text": sentence_text,
                "tokens": [{
                    "word": token.find("word").text,
                    "lemma": token.find("lemma").text,
                    "POS": token.find("POS").text,
                    "NER": token.find("NER").text
                } for token in sentence.findall(".//token")]
            }

            dependencies = []
            for dep in sentence.findall(".//basic-dependencies/dep"):
                dep_type = dep.get("type")
                governor_idx = dep.find("governor").text
                dependent_idx = dep.find("dependent").text
                dependencies.append({
                    "dep_type": dep_type,
                    "governor": governor_idx,
                    "dependent": dependent_idx
                })
            sentence_data["dependencies"] = dependencies

            sentences_data.append(sentence_data)

        coreferences = []
        for coref_chain in root.findall(".//coreference"):
            coref_chain_data = []
            for mention in coref_chain.findall("mention"):
                coref_data = {
                    "representative": mention.get("representative") == "true",
                    "sentence_id": mention.find("sentence").text,
                    "start": int(mention.find("start").text),
                    "end": int(mention.find("end").text),
                    "head": int(mention.find("head").text)
                }
                coref_chain_data.append(coref_data)
            coreferences.append(coref_chain_data)

    return {"sentences_data": sentences_data, "coreferences": coreferences}

def filter_sentences_by_character(character_name, sentences_data, coreferences_data):
    character_sentences = []

    for sentence_data in sentences_data:
        if character_name.lower() in sentence_data["sentence_text"].lower():
            character_sentences.append(sentence_data["sentence_text"])

    for coref_chain in coreferences_data:
        representative_mention = next((m for m in coref_chain if m["representative"]), None)
        if representative_mention:
            sentence_id = representative_mention["sentence_id"]
            sentence_text = next(
                (s["sentence_text"] for s in sentences_data if s["sentence_id"] == sentence_id),
                None
            )
            if sentence_text and character_name.lower() in sentence_text.lower():
                for mention in coref_chain:
                    sentence_data = next(
                        (s for s in sentences_data if s["sentence_id"] == mention["sentence_id"]),
                        None
                    )
                    if sentence_data:
                        character_sentences.append(sentence_data["sentence_text"])
                        
    character_sentences = list(set(character_sentences))

    df = pd.DataFrame({
        "character_sentences": character_sentences
    })
    return df

def get_sentiment_with_textblob(sentence):
    analysis = TextBlob(sentence)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def process_character_sentiments_textblob(sentences):
    polarity_textblob = []
    subjectivity_textblob = []
    
    for sentence in sentences:
        polarity, subjectivity = get_sentiment_with_textblob(sentence)
        polarity_textblob.append(polarity)
        subjectivity_textblob.append(subjectivity)
    
    mean_polarity_textblob = np.mean(polarity_textblob) if polarity_textblob else None
    mean_subjectivity_textblob = np.mean(subjectivity_textblob) if subjectivity_textblob else None

    return mean_polarity_textblob, mean_subjectivity_textblob

def interpret_polarity(p):
    if -1 <= p < -0.5:
        return "Very bad guy"
    elif -0.5 <= p < 0:
        return "Bad guy"
    elif p == 0:
        return "Neutral"
    elif 0 < p <= 0.5:
        return "Nice guy"
    elif 0.5 < p <= 1:
        return "Very nice guy"
    
def good_guy_detector(polarity):
    if -0.25 <= polarity <= 0.05:
        return "Not significant"
    elif polarity > 0.05:
        return 1
    else:
        return 0
    
class GoodBadGuyAnalyzer(Analyzer):
    def __init__(self, data):
        super().__init__(data)
        self.genres_list = ['Action & Adventure', 'Drama', 'Comedy', 'Horror & Thriller', 
              'Fantasy & Sci-Fi', 'Historical & War', 'Romance', 'Documentary', 
              'Music & Performance', 'Cult & B-Movies', 'Other']
        self.data["Genre_Category"] = self.data['Genres'].apply(lambda x: self._categorize_genre(x))

    def _categorize_genre(self,genres_movies) -> list:
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

        for genre in genres_movies:
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

    def get_vowel_stats(self):
        vowels = set('aeiouy')

        def count_vowels(name):
            return sum(1 for char in name.lower() if char in vowels)

        def count_consonants(name):
            return sum(1 for char in name.lower() if char not in vowels)

        self.data['vowel_count'] = self.data['Character_name'].apply(count_vowels)
        self.data['consonant_count'] = self.data['Character_name'].apply(count_consonants)

        stats_kindness_vowels = self.data.groupby('kindness')['vowel_count'].agg(['mean', 'std'])
        stats_kindness_consonants = self.data.groupby('kindness')['consonant_count'].agg(['mean', 'std'])
        return stats_kindness_vowels, stats_kindness_consonants

    def get_length_stats(self):
        self.data['name_length'] = self.data['Character_name'].apply(lambda name: len(name))
        stats_length = self.data.groupby('kindness')['name_length'].agg(['mean', 'std'])

        return stats_length

    def create_boxenplot_by_sex(self):
        self.data['name_length'] = self.data['Character_name'].apply(lambda name: len(name))

        fig = sns.boxenplot(self.data, x='kindness', y='name_length')
        fig.set_xlabel('Kindness')
        fig.set_ylabel('Name Length')
        fig.set_title('Name Length per Kindness')

    def get_vowel_percentage(self):
        self.data['vowel_percentage'] = self.data['vowel_count'] / self.data['name_length']
        percent_vowels = self.data.groupby('kindness')['vowel_percentage'].agg(['mean', 'std'])

        return percent_vowels

    def _create_letter_count_df(self, letter_position):

        df_letter = self.data.copy()
        df_letter['letter'] = df_letter['Character_name'].apply(lambda name: name[letter_position].lower())
        
        letter_counts_H = df_letter[df_letter['kindness'] == 'M']['letter'].value_counts()
        letter_counts_F = df_letter[df_letter['kindness'] == 'F']['letter'].value_counts()

        male_count = df_letter[df_letter['kindness'] == 'M'].shape[0]
        female_count = df_letter[df_letter['kindness'] == 'F'].shape[0]
        letter_counts_H_percentage = letter_counts_H / male_count*100
        letter_counts_F_percentage = letter_counts_F / female_count*100
        letter_counts = pd.concat([letter_counts_H_percentage, letter_counts_F_percentage], axis=1)
        letter_counts.columns = ['letter_men', 'letter_women']
        letter_counts = letter_counts.head(26)  # Limit to top 26 letters

        top_letter_names = (
            df_letter.groupby(['letter', 'kindness'])['Character_name']
            .apply(lambda x: x.value_counts().head(3).index.tolist())
            .unstack(fill_value=[])
        )

        return letter_counts, top_letter_names

    def plot_letter_name_percentage(self, letter_position):

        letter_counts, top_letter_names = self._create_letter_count_df(letter_position)

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

    def plot_age_kindness_distribution_with_top_names(self):
        age_bins = [0, 12, 17, 24, 34, 44, 54, 64, 74, 84, 100]
        age_labels = [
            '<12y', '13y-17y', '18y-24y', '25y-34y', '35y-44y', 
            '45y-54y', '55y-64y', '65y-74y', '75y-84y', '>85y'
        ]

        self.data['age_category'] = pd.cut(self.data['Actor_age'], bins=age_bins, labels=age_labels, right=False)

        age_kindess_counts = self.data.groupby(['age_category', 'kindness'], observed= True).size().unstack(fill_value=0)
        total_counts = self.data['kindness'].value_counts()
        age_kindness_percentage = age_kindness_counts.div(total_counts, axis=1) * 100

        top_names = (
            self.data.groupby(['age_category', 'kindness'], observed= True)['Character_name']
            .apply(lambda x: x.value_counts().head(3).index.tolist())
            .unstack(fill_value=[])
        )
        fig = go.Figure()

        for kindness in [0, 1]:
            fig.add_trace(go.Bar(
                x=age_labels,
                y=age_kindness_percentage[kindness],
                name='Male' if kindness == 'M' else 'Female',
                marker_color='skyblue' if kindness == 'M' else 'salmon',
                hovertext=[f"Top names: {', '.join(top_names.loc[age, kindness])}" for age in age_labels],
                hoverinfo="text"
            ))
        fig.update_layout(
            title='Percentage of Good guys and Bad guys in Each Age Category',
            xaxis_title='Age Category',
            yaxis_title='% of Total Good guys/Bad guys',
            barmode='group',
            xaxis=dict(tickvals=age_labels, tickangle=0),
            yaxis=dict(ticksuffix='%'),
            legend=dict(title="Gender")
        )

        fig.show()

    def get_good_guys_girls_names_SA(self, nb_of_names):
        genres = self.genres_list
        df_male = self.data[self.data['Sex'] == 'M']
        df_female = self.data[self.data['Sex'] == 'F']

        top_male_names_by_genre = {}
        top_female_names_by_genre = {}

        # We loop through each genre to get top names for males and females
        for genre in genres:
            male_genre_names = df_male[df_male['Genre_Category'].apply(lambda categories: genre in categories)]
            female_genre_names = df_female[df_female['Genre_Category'].apply(lambda categories: genre in categories)]

            # Sort by Polarity ascending and select the lowest
            top_male_names = male_genre_names.nlargest(nb_of_names, 'Polarity')['Character_name'].tolist()
            top_male_names_by_genre[genre] = top_male_names

            top_female_names = female_genre_names.nlargest(nb_of_names, 'Polarity')['Character_name'].tolist()
            top_female_names_by_genre[genre] = top_female_names

        good_guys_names_m = pd.DataFrame.from_dict(top_male_names_by_genre, orient='index').transpose()
        good_girls_names_f = pd.DataFrame.from_dict(top_female_names_by_genre, orient='index').transpose()

        return good_guys_names_m, good_girls_names_f
    
    def get_bad_guys_girls_names_SA(self, nb_of_names):
        genres = self.genres_list
        df_male = self.data[self.data['Sex'] == 'M']
        df_female = self.data[self.data['Sex'] == 'F']

        top_male_names_by_genre = {}
        top_female_names_by_genre = {}

        # We loop through each genre to get top names for males and females
        for genre in genres:
            male_genre_names = df_male[df_male['Genre_Category'].apply(lambda categories: genre in categories)]
            female_genre_names = df_female[df_female['Genre_Category'].apply(lambda categories: genre in categories)]

            # Sort by Polarity ascending and select the lowest
            top_male_names = male_genre_names.nsmallest(nb_of_names, 'Polarity')['Character_name'].tolist()
            top_male_names_by_genre[genre] = top_male_names

            top_female_names = female_genre_names.nsmallest(nb_of_names, 'Polarity')['Character_name'].tolist()
            top_female_names_by_genre[genre] = top_female_names

        bad_guys_names_m = pd.DataFrame.from_dict(top_male_names_by_genre, orient='index').transpose()
        bad_girls_names_f = pd.DataFrame.from_dict(top_female_names_by_genre, orient='index').transpose()

        return bad_guys_names_m, bad_girls_names_f


    def create_sunburst_data_SA(self,frequent_names_f):
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
