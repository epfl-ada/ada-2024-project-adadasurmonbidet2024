import os
import pandas as pd

def remove_nan_or_space_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove rows with NaN values or rows where the specified column contains only spaces.
    """
    return df.dropna(subset=[column]).loc[df[column].str.strip() != '']

def select_first_word(name):
    return name.split()[0] if name.split() else None

invalid_chars = set("-/.,'\"#()0123456789")
vowels = set("aeyuio")

russian_set = {'USSR', 'russian', 'russia'}

def filter_names(name):
    words_in_name = name.split()

    filtered_words = [
        word for word in words_in_name 
        if all(char not in word for char in invalid_chars)
        and any(char in vowels for char in word.lower())
        and len(word)>1
    ]
    return ' '.join(filtered_words)

def filter_country_names(country: str) -> str:
    """
    Normalize country names and assign them to regions while keeping unclassified names as-is.
    """
    # Normalize country names
    normalized_country = country.strip().lower()

    # Explicit normalization rules for alternative/historical names
    normalization_map = {
        "ussr": "ex-ussr",
        "unified team (ex ussr)": "ex-ussr",
        "russian federation": "russia", 
        'federal republic of germany (1950-1990, "ger" since)': "germany",
        "federal republic of germany (1950-1990, 'ger' since)": "germany",
        'federal republic of germany (1950-1990, ""ger"" since)': "germany",
        "german democratic republic (1955-1990," : "germany",
        "german democratic republic (1955-1990)": "germany",  
        "democratic people's republic of korea": "korea",  
        "united team of germany (1956,1960,1964)": "germany",
        "czechoslovakia": "czech republic",
        "serbia and montenegro": "serbia",
        "the former yugoslav republic of macedonia": "macedonia",
        "yugoslavia": "serbia",
        "people's republic of china": "china",
        "republic of korea": "korea",
        "islamic republic of iran": "iran",
        "great britain": "uk",
        "united states of america": "usa",
    }

    normalized_country = normalization_map.get(normalized_country, normalized_country)

    region_mappings = {
        "Germanic": ["germany", "austria", "switzerland", "luxembourg", "liechtenstein", "netherlands"],
        "Romance": ["france", "belgium", "italy", "andorra", "romania", "moldova"],
        "Slavic": ["russia", "ukraine", "belarus", "poland", "czech republic", "slovakia", "bulgaria", "serbia", "croatia", "bosnia and herzegovina", "montenegro", "macedonia", "slovenia", "kazakhstan", "uzbekistan", "armenia", "georgia", "azerbaijan"],
        "English-Speaking": ["uk", "usa", "canada", "ireland", "australia", "new zealand"],
        "Baltic": ["lithuania", "latvia", "estonia"],
        "Hispanic": ["spain", "mexico", "argentina", "brazil", "portugal", "chile", "colombia", "peru", "ecuador", "venezuela", "paraguay", "uruguay", "cuba", "jamaica", "trinidad and tobago"],
        "Nordic": ["denmark", "sweden", "norway", "finland", "iceland"],
        "Middle Eastern": ["iran", "turkey", "israel", "iraq", "syria", "lebanon", "jordan", "saudi arabia", "united arab emirates"],
        "South Asian": ["india", "pakistan", "bangladesh", "nepal", "sri lanka", "maldives", "bhutan"],
        "East Asian": ["china", "japan", "korea", "taiwan", "mongolia"],
        "Southeast Asian": ["thailand", "vietnam", "philippines", "malaysia", "indonesia", "cambodia", "laos", "myanmar"],
        "North Africa": ["egypt", "morocco", "algeria", "tunisia", "libya", "sudan"],
        "Sub-Saharan Africa": ["nigeria", "kenya", "south africa", "ghana", "ethiopia", "angola", "uganda", "zambia", "senegal", "togo", "zimbabwe"],
        "Oceania": ["fiji", "papua new guinea", "samoa", "tonga"],
    }

    # Assign to a region or return the normalized country as-is
    for region, countries in region_mappings.items():
        if normalized_country in countries:
            return region
    return None  # Keep the original normalized country if no match is found

def filter_countries_with_min_names(df: pd.DataFrame, min_names=400) -> pd.DataFrame:
    country_counts = df['Country'].value_counts()
    valid_countries = country_counts[country_counts >= min_names].index
    return df[df['Country'].isin(valid_countries)]

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

file_path_1 = os.path.join(script_dir, '..', 'new_dataset', 'test.txt')
file_path_2 = os.path.join(script_dir, '..', 'new_dataset', 'train.txt')

# Read the files
df_ethny_1 = pd.read_csv(file_path_1, sep='\t', names=['Name', 'Country'])
df_ethny_2 = pd.read_csv(file_path_2, sep='\t', names=['Name', 'Country'])

# Concatenate in one dataframe
df_ethny = pd.concat([df_ethny_1, df_ethny_2], ignore_index=True)

# Filter the names
df_ethny['Name'] = df_ethny['Name'].apply(select_first_word)
df_ethny['Name'] = df_ethny['Name'].apply(filter_names)

df_ethny = remove_nan_or_space_rows(df_ethny,'Name')

# Filter the countries
df_ethny['Country'] = df_ethny['Country'].apply(filter_country_names)

df_ethny = filter_countries_with_min_names(df_ethny, min_names=400)

# Reset index to ensure a clean output
df_ethny = df_ethny.reset_index(drop=True)

# Export the dataset in CSV file
print('The cleaned dataset contains', df_ethny.shape[0], 'rows and', df_ethny.shape[1], 'columns.')

df_ethny.to_csv('data/name_ethnicity.csv', index=False)