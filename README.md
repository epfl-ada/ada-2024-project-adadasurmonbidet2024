# [ADAdasurmonbidet2024 - DISCOVER CHARACTERS' FEATURES BASED ON FIRST NAME](https://fannymissillier.github.io)

Link to the Data Story: 

## Project Idea
In film storytelling, a character’s name is rarely just a label; it often serves as a subtle cue to their personality and role in the movie. Our project tackles the intriguing question: **"Can a character’s archetype be predicted from their name ?"** Through this data analysis project, we aim to decode connections between specific name characteristics — such as length, structure and phonetics — and the characters' features including gender, age, the movie genre, its origin and even its morality. Additionally, we will investigate how naming conventions differ across contexts, with a particular focus on comparing movie productions from the United States. Specifically, we aim to identify strong correlations between the portrayal of "nice guys" and "bad guys," focusing on potential differences in the depiction of Soviet characters versus American characters.

![Names' Features](data/Image/intro.png)

## Research Questions

About discovering characters' features based on their first name: 
- Is there a correlation between a characters' name and their **gender** ?  
- Can we observe a relationship between a character's name and their **age** ?  
- How does a character's **origin** influence their name ?  
- Are character names linked to the **movie genre** they appear in ?  
- Is there a pattern in character names that distinguishes **Nice guys** from **Bad guys** in movies ?  

About how naming conventions differ across contexts: 
- How do naming conventions in U.S. movie productions during the Cold War reflect ideological narratives, particularly in the portrayal of Soviet characters as "good guys" or "bad guys" ?  


## Datasets

### CMU Movie Summary Corpus
The **CMU Movie Summary Corpus** is a dataset provides a detailed view of over 42,000 movie plot summaries, extracted from Wikipedia and aligned with metadata from Freebase:

- **Movie Metadata**:
  - Box office revenue
  - Genre
  - Release date
  - Runtime
  - Language

- **Character Information**:
  - Names of characters
  - Metadata aligned with the actors who portray them, including:
    - Gender
    - Estimated age at the time of the movie's release

- **Processed Data Supplement**:
  - Stanford CoreNLP-processed summaries, which include tagging, parsing, Named Entity Recognition (NER), and coreference resolution.

**Source**: [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/)

---

### Name Ethnicity Dataset
The **Name Ethnicity Dataset**, sourced from Kaggle, provides a collection of names alongside their corresponding ethnic origins. 

**Dataset Features**:
- **Name Data**:
  - First and last names
- **Associated Metadata**:
  - Country of origin
  - Ethnicity information

**Source**: [Name Ethnicity Dataset on Kaggle](https://www.kaggle.com/datasets/tommylariccia/name-ethnicity-data)



## Project Plans and Methods

### Task 0: Data Exploration
We began with data cleaning, removing irrelevant columns and rows, particularly those missing the character name. For preprocessing, we isolated first names by:
- Stripping out prefixes (e.g., “Dr. Alison Parker” becomes “Alison Parker”)
- Eliminate commun names like “taxi” or “waiter” using `nltk.corpus.words`. 
- From the filtered data, restore words identified as valid names using the `nltk.corpus.names` library, which provides a list of English names. 
- Keeping only the first name from full names (e.g., “Alison” from “Alison Parker")

The cleaned data was saved in the `cleaned.csv` file for streamlined analysis.

### Task 1: Present globally the data

### Task 2: Feature Extraction for Name Analysis
We aim to analyze gender and ethnicity, as they are the only two fixed character traits that remain independent of contextual factors such as the production country or movie genre.
We examined the name characteristics by gender to justify the division of male and female for the rest of the analysis. We analysed several characteristics of the names :
- Structure : 
    - length using Python’s `len()` function
    - first and last letters
    - vowel and consonant count using `pandas`
    - n-gram (first test, will be done for P3)
- Phonetics (P3, optional) using algorithms like Soundex or Metaphone to explore pronunciation patterns and trends related to names

We proved significant differences in the structure of female/male names.

**For P3 :** We'll add (using an external dataset: "Name_Ethnicity") an analysis on: Ethnicity. This aims to compare the origins of character names with the country of production of the movie and the roles assigned to these characters, revealing cultural patterns or contrasts in naming and characterization within films.
We'll be looking for differences in structure and/or phonetic of name from different ethnicity. 

### Task 3: Understanding the Data
To better understand the relationships between character names and movies, we started by looking at the data to identify patterns and formulate hypotheses. Our goal was to find trends in character features, such as the distribution of names across different movie genres, countries and ages.

### Task 4: Study of Trends Between Name Characteristics and Character Features
We identified statistically significant associations between names and movie genre/country.
We plan on identifying trends in the data, especially between name characteristics and character features. We will examine whether certain sounds are tied to specific genres or character roles and explore the influence of filming locations on name origins.

### Task 5: Sentiment Analysis
The goal is to analyze the connotations of character names and how they are linked to name characteristics. We will conduct sentiment analysis on sentences featuring these names, identifying structures and adjectives to determine if the names have positive, negative, or neutral connotations. Phonetic analysis of names will further explore whether phonetic traits (like name length or consonant count) correlate with specific roles, such as villains.

### Task 6: Predictive Model & Generative List of Heroes/Villains
We aim to develop a “Character ID” model that predicts the most likely characteristics of a movie character based on their first name. The model would generate a character profile, including attributes such as gender, ethnicity, age, movie genre and whether the character is a villain or hero.

### Task 7: Create Data Story
Create the website.

## Organisation:
- Luca: Task 6
- AInhoa: Task 2 and 4
- Fanny: Task 5 and 7
- Zacharie: Task 4 and 6
- Amaury: Task 2 and 4

Can be rearanged if some Tasks are longer than others.

## Proposed Timeline
- *22.11.2023*: Task 1 to 3 completely done, work on Task 5
- *29.11.2024*: Homework 2
- *06.12.2023*: Task 4 done, start Task 6, finalize Task 5
- *13.12.2023*: Task 6 and 7
- *20.12.2023*: Deadline P3