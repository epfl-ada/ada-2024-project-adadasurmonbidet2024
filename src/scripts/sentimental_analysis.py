import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import gzip
import xml.etree.ElementTree as Xet
from textblob import TextBlob
import nbformat
from utils.results_utils import *

df_char_cleaned = pd.read_csv('data/cleaned.csv', sep=',', header=0)

character_names_by_film = df_char_cleaned.groupby("Wikipedia_ID")["Character_name"].unique().reset_index()  

base_dir='data/corenlp_plot_summaries'
movie_ids = character_names_by_film["Wikipedia_ID"]

data = []
count = 0 
for index, movie_id in enumerate(movie_ids):
    file_path = os.path.join(base_dir, f"{movie_id}.xml.gz") 
    if os.path.exists(file_path):
        movie_nlp = parse_xml_gz(file_path)
        list_characters = character_names_by_film["Character_name"].iloc[index]
        for character_name in list_characters:
            filtered_sentences = filter_sentences_by_character(character_name, movie_nlp["sentences_data"], movie_nlp["coreferences"])
            character_sentences = []

            for sentence in filtered_sentences["character_sentences"]:
                character_sentences.append(sentence) 
                count += 1
                print(count/650000)
            data.append({
                "Character_Name": character_name,
                "Wikipedia_ID": movie_id,
                "Sentences": character_sentences,
                })
            
df = pd.DataFrame(data)
df = df[df['Sentences'].apply(lambda x: len(x) > 0)]
df['Sentence_Count'] = df['Sentences'].apply(len)

data_textblob = []  
list_characters = df["Character_Name"]
count = 0

for index, character_name in enumerate(list_characters):
    list_sentences_by_character = df["Sentences"].iloc[index]
    movie_id = df["Wikipedia_ID"].iloc[index]
    count+=1
    print(count)
    mean_polarity_textblob, mean_subjectivity_textblob = process_character_sentiments_textblob(list_sentences_by_character)
    data_textblob.append({
        "Character_Name": character_name,
        "Wikipedia_id": movie_id,
        "Polarity": mean_polarity_textblob,
        "Subjectivity" : mean_subjectivity_textblob
    })

df_sentiment_analysis_textblob = pd.DataFrame(data_textblob)

df_sentiment_analysis_textblob.to_csv('data/sentimental_analysis.csv', index=False)