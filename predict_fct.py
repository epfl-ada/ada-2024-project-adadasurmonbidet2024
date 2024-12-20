import pandas as pd
import pickle
from src.utils.ml_utils import *
from sklearn.preprocessing import MultiLabelBinarizer

### Loading of the predictive models

# Path to the different models
path_genre = 'predictive_model/model_genres.pkl'
path_gender = 'predictive_model/model_Sex.pkl'
path_age = 'predictive_model/model_age_category.pkl'
path_kindness = 'predictive_model/model_kindness.pkl'
path_origin = 'predictive_model/model_Country.pkl'

# Load the models
with open(path_genre, 'rb') as file:
    predict_genres = pickle.load(file)

with open(path_gender, 'rb') as file:
    predict_gender = pickle.load(file)

with open(path_age, 'rb') as file:
    predict_age = pickle.load(file)

with open(path_kindness, 'rb') as file:
    predict_kindness = pickle.load(file)

with open(path_origin, 'rb') as file:
    predict_origin = pickle.load(file)

augmented_alphabet = 'abcdefghijklmnopqrstuvwxyzéèíá'

### Prediction of the movie genre

genres_list = ['Action & Adventure', 'Drama', 'Comedy', 'Horror & Thriller', 
              'Fantasy & Sci-Fi', 'Historical & War', 'Romance', 'Documentary', 
              'Music & Performance', 'Cult & B-Movies', 'Other']

# Apply MultiLabelBinarizer to encode the genres
mlb = MultiLabelBinarizer(classes=genres_list)
mlb.fit([genres_list])

def feature_creation_g(name):
    df_pred = pd.DataFrame([name], columns=['Name'])
    pred_processor = NameFeatureProcessor('Name', ngram_range=(2,2))
    df_pred = pred_processor.process(df_pred, alphabet=augmented_alphabet, analyze_name=True, diacritic=False, phonetics=False, first_last=True, ngram=False)
    
    # Load pre-trained HashingVectorizer
    with open('predictive_model/hashing_vectorizer_genre.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Add ngram using the vectorizer
    ngram_name = vectorizer.transform(df_pred['Name'])
    ngram_name_df = pd.DataFrame(ngram_name.toarray())
    df_pred = pd.concat([df_pred, ngram_name_df], axis=1)
    
    return df_pred

def predict(df, pred_model):
    df.drop(columns=['Name'], inplace=True)
    df.columns = df.columns.astype(str)  # Ensure column names are strings
    return pred_model.predict(df)

def create_and_predict_genre(name, model):
    df = feature_creation_g(name)
    pred = predict(df, model)
    
    # Decode the binary prediction results to genre names
    decoded_genres = mlb.inverse_transform(pred)

    print(decoded_genres)
    return decoded_genres

### Prediction of the gender/ age/ kindness

# Adapted fct for slightly different model 
def feature_creation(name):
    augmented_alphabet = 'abcdefghijklmnopqrstuvwxyzéèíá'

    df_pred = pd.DataFrame([name], columns=['Name'])
    pred_processor = NameFeatureProcessor('Name',ngram_range=(2,2))
    df_pred =pred_processor.process(df_pred,alphabet = augmented_alphabet, analyze_name = True, diacritic = False, phonetics = False, first_last = True, ngram=False)

    with open(f'predictive_model/hashing_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    ngram_name = vectorizer.transform(df_pred['Name'])
    ngram_name_df = pd.DataFrame(ngram_name.toarray())
    df_pred = pd.concat([df_pred, ngram_name_df], axis=1)
    return df_pred

def create_and_predict(name, model):
    df = feature_creation(name)
    pred = predict(df, model)

    print(pred)
    return pred

### Prediction of the origin

# Adapted fct for slightly different model 
def feature_creation_o(name):
    augmented_alphabet = 'abcdefghijklmnopqrstuvwxyzéèíá'

    df_pred = pd.DataFrame([name], columns=['Name'])
    pred_processor = NameFeatureProcessor('Name',ngram_range=(2,3))
    df_pred =pred_processor.process(df_pred,alphabet = augmented_alphabet,analyze_name = True, diacritic = False, phonetics = False, first_last = True, ngram=False)

    with open(f'predictive_model/hashing_vectorizer_origin.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    ngram_name = vectorizer.transform(df_pred['Name'])
    ngram_name_df = pd.DataFrame(ngram_name.toarray())
    df_pred = pd.concat([df_pred, ngram_name_df], axis=1)
    return df_pred
    
def create_and_predict_origin(name, model):
    df = feature_creation_o(name)
    pred = predict(df, model)
    print(pred)
    return pred

### Print all the predictions for a chosen name
def prediction_fct(name = 'José'):
    genres = create_and_predict_genre(name, predict_genres)
    gender = create_and_predict(name, predict_gender)
    age = create_and_predict(name, predict_age)
    kindness = create_and_predict(name, predict_kindness)
    origin = create_and_predict_origin(name, predict_origin)
    return genres, gender, age, kindness, origin

prediction_fct('Fanny')