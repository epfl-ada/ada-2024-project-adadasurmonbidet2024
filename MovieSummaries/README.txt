# ADAdasurmonbidet2024 - DISCOVER CHARACTERS' FEATURES BASED ON FIRST NAME

### Project Ideas
In film storytelling, a character’s name is rarely just a label; it often serves as a subtle cue to their personality and role in the movie. Our project tackles the intriguing question: **“Can a character’s archetype be predicted from their name?”** Through this data analysis project, we aim to decode connections between specific name characteristics — such as length, structure, cultural origin, and phonetics — and traits including gender, narrative role, movie genre-specific role, and additional properties that we expect to uncover through deeper analysis. A sentiment analysis will also allow us to explore whether these name characteristics evoke positive or negative associations, potentially correlating with archetypes such as hero, villain, or more nuanced characters, if time permits.

![Names' Features](MovieSummaries/Image_data.png)

### Research Questions
- What are the key predictors of character traits based on first names?
- Can the number of vowels, consonants, or specific sounds in a name correlate with the character’s role (e.g., hero, villain)?
- How do the length and structure of a character’s first name correlate with their role in the narrative? Do longer or more complex names correspond to more prominent or villainous roles in films?
- Can sentiment analysis of names offer insights into character traits?
- Do films produced in specific regions tend to use names that align with the cultural background of the characters? For example, do American films predominantly feature Anglo-Saxon names for protagonists and villainous roles?

### Additional Dataset
We propose integrating a dataset that associates each first name with its ethnicity, which would allow us to make the ethnic origin of the names one of the main characteristics in our analysis. This would enable us to, for example, explore the relationship between the origin of a name and the location where the films are produced. The dataset we have identified, “Name Ethnicity,” is available on Kaggle.

### Project Plans and Methods

#### Step 1: Data Exploration
We began with data cleaning, removing irrelevant columns and rows, particularly those missing character names. For preprocessing, we isolated first names by:
- Stripping out prefixes (e.g., “Dr. Alison Parker” becomes “Alison”)
- Removing generic terms like “girlfriend”
- Keeping only the first name from full names (e.g., “Amélie” from “Amélie Poulain”)

The cleaned data was saved in a CSV file for streamlined analysis.

#### Step 2: Feature Extraction for Name Analysis
In this step, we examined the characteristics of first names to lay the foundation for the rest of our analysis. We extracted several specific traits of the names:
- Length (using Python’s `len()` function)
- Structure (vowel and consonant count using pandas)
- First and last letters

Next, we will merge the cultural origin dataset and apply phonetic analysis using algorithms like Soundex or Metaphone to explore pronunciation patterns and trends related to names.

#### Step 3: Understanding the Data
To better understand the relationships between character names and their traits, we started by exploring the data to identify patterns and formulate hypotheses. Our goal is to uncover trends related to gender, age, movie genres, character importance, ethnicity, and the phonetic characteristics of names.

#### Step 4: Study of Correlations Between Name Characteristics and Character Features
We focus on identifying key correlations in the data, especially phonetic patterns linked to factors like gender, movie genre, age, and ethnicity. We will examine whether certain sounds are tied to specific genres or character roles and explore the influence of filming locations on name origins.

#### Step 5: Sentiment Analysis
The goal is to analyze the connotations of character names and how they are linked to character traits. We will conduct sentiment analysis on sentences featuring these names, identifying structures and adjectives to determine if the names have positive, negative, or neutral connotations. Phonetic analysis will further explore whether phonetic traits (like name length or consonant count) correlate with specific roles, such as villains.

#### Step 6: Predictive Model & Generative List of Heroes/Villains
Our aim is to develop a “Character ID” model that predicts the most likely characteristics of a movie character based on their first name. The model would generate a character profile, including attributes such as age, ethnicity, gender, and whether the character is a villain or hero. Additionally, we plan to generate a list of first names most commonly associated with villains or heroes in films.

#### Step 7: Create Data Story
We will summarize and visualize our findings in an engaging narrative, presenting the insights gained from the analysis.

### Organisation:
COMPLETER

### Proposed Timeline
- **22.11.2023**: Step 1 to 3
- **29.11.2024**: Homework 2
- **13/12/2023**: Step 4 to 5
- **18.12.2023**: Step 6 to 7
- **20.12.2023**: Deadline Milestone 3
