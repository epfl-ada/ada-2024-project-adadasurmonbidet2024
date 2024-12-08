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

def filter_country_names(country):
    """
    Filters and unifies country names
    """
    country_lower = country.lower()

    if any(keyword in country_lower for keyword in ['ussr', 'russian', 'russia']):
        return 'ex-USSR'
    elif 'germany' in country_lower:
        return 'Germany'
    elif 'korea' in country_lower:
        return 'Korea'
    elif any(keyword in country_lower for keyword in ['united states of america', 'great britain', 'australia', 'new zealand', 'canada']):
        return 'British'
    elif any(keyword in country_lower for keyword in ['france', 'belgium', 'luxembourg']):
        return 'France/Belgium'
    elif any(keyword in country_lower for keyword in ['spain', 'argentina', 'colombia', 'chile', 'uruguay', 'paraguay', 'peru', 'venezuela']):
        return 'Hispanic'
    else:
        return country

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

#print(df_ethny.sample(10))

# Filter the countries

df_ethny['Country'] = df_ethny['Country'].apply(filter_country_names)

country_count = df_ethny['Country'].value_counts()
#print(country_count.head(10))

# Export the dataset in csv file

print('The cleaned dataset contains',df_ethny.shape[0],'rows and',df_ethny.shape[1],'columns')

df_ethny.to_csv('data/name_ethnicity.csv', index=False)
