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
invalid_chars = set("-/.,'\"#()0123456789Ú'´'́':!\’Æ“-~;[Ş”`]-с")
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
        if word.lower() not in english_words                    # Exclude English words
        and word.lower() not in invalid_word_list               # Exclude invalid words
        and all(char not in word for char in invalid_chars)     # Exclude words with invalid characters
        and any(char in vowels for char in word.lower())        # Ensure the word contains at least one vowel
        and sum(1 for char in word if char.isupper()) <= 1      # Limit uppercase letters to 1
        and word.istitle()                                      # Ensure the word starts with a capital letter
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


class GenreCategorizer:
    def __init__(self):
        # Define genre categories
        self.action_adventure = ['Action', 'Adventure', 'Thriller', 'War film', 'Action/Adventure', 'Martial Arts Film', 'Wuxia', 'Superhero movie', 'Western', 'Sword and sorcery', 'Spy', 'Supernatural']
        self.drama = ['Drama', 'Biographical film', 'Crime Drama', 'Family Film', 'Family Drama', 'Historical fiction', 'Biopic [feature]', 'Courtroom Drama', 'Political drama', 'Family-Oriented Adventure', 'Psychological thriller']
        self.comedy = ['Comedy', 'Romantic comedy', 'Comedy-drama', 'Comedy film', 'Black comedy', 'Slapstick', 'Romantic comedy', 'Musical', 'Satire', 'Parody', 'Comedy horror']
        self.horror_thriller = ['Horror', 'Psychological horror', 'Horror Comedy', 'Slasher', 'Thriller', 'Crime Thriller', 'Sci-Fi Horror', 'Suspense', 'Zombie Film', 'Natural horror films']
        self.fantasy_sci = ['Fantasy', 'Science Fiction', 'Space western', 'Fantasy Adventure', 'Fantasy Comedy', 'Sci-Fi Horror', 'Sci-Fi Thriller', 'Fantasy Drama', 'Dystopia', 'Alien Film', 'Cyberpunk', 'Time travel']
        self.historical_war = ['Historical drama', 'Historical fiction', 'Historical Epic', 'Epic', 'War effort', 'War film', 'Period piece', 'Courtroom Drama']
        self.romance = ['Romance Film', 'Romantic drama', 'Romance', 'Romantic fantasy', 'Marriage Drama']
        self.documentary = ['Documentary', 'Docudrama', 'Biography', 'Historical Documentaries', 'Mondo film', 'Patriotic film', 'Educational']
        self.music_performance = ['Musical', 'Music', 'Musical Drama', 'Musical comedy', 'Dance', 'Jukebox musical', 'Concert film']
        self.cult_b_movies = ['Cult', 'B-movie', 'Indie', 'Experimental film', 'Surrealism', 'Avant-garde', 'Grindhouse', 'Blaxploitation', 'Camp']

    def _categorize_genre(self, genres_movies) -> list:
        categories = []
        
        # Iterate through the genres and categorize
        for genre in genres_movies:
            if genre in self.action_adventure:
                if 'Action & Adventure' not in categories:
                    categories.append('Action & Adventure')
            if genre in self.drama:
                if 'Drama' not in categories:
                    categories.append('Drama')
            if genre in self.comedy:
                if 'Comedy' not in categories:
                    categories.append('Comedy')
            if genre in self.horror_thriller:
                if 'Horror & Thriller' not in categories:
                    categories.append('Horror & Thriller')
            if genre in self.fantasy_sci:
                if 'Fantasy & Sci-Fi' not in categories:
                    categories.append('Fantasy & Sci-Fi')
            if genre in self.historical_war:
                if 'Historical & War' not in categories:
                    categories.append('Historical & War')
            if genre in self.romance:
                if 'Romance' not in categories:
                    categories.append('Romance')
            if genre in self.documentary:
                if 'Documentary' not in categories:
                    categories.append('Documentary')
            if genre in self.music_performance:
                if 'Music & Performance' not in categories:
                    categories.append('Music & Performance')
            if genre in self.cult_b_movies:
                if 'Cult & B-Movies' not in categories:
                    categories.append('Cult & B-Movies')

        return categories if categories else ['Other']

    def categorize_genres_in_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Apply genre categorization to the 'genre' column and create a new 'categorized_genre' column
        df['Genre_Category'] = df['Genres'].apply(self._categorize_genre)
        return df
