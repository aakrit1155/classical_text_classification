import os
# Data manipulation
#===========================================================================
import pandas as pd
import numpy as np

# Preprocessing
#===========================================================================
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Models
#===========================================================================
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Metrics
#===========================================================================
from sklearn.metrics import confusion_matrix, accuracy_score

# tqdm
#===========================================================================
from tqdm.auto import tqdm
from . import plots

nlp = spacy.load("en_core_web_sm")

SEED = 43
file_path = "./data/df_file.csv"
figures_path = "./figures"

def load_data(fp=file_path):
  df = pd.read_csv(fp)
  print(f'Rows: {df.shape[0]}')
  print(f'Columns: {df.shape[1]}')
  print("--------------------------------------------")
  print(" " * 10,"Dataset Information")
  print("--------------------------------------------")
  print(df.info())
  # The data had some duplicates so I will be removing them down below:
  df = df.drop_duplicates(ignore_index = True)
  print("Duplicates after removing: ", df.duplicated().sum())
  # If the data had null values, we would have to handle them too. 
  # However, luckily, the data that we are working with has no null values
  return df

def create_figures_dir(fig_path=figures_path):
  """
  Create Directory to save plots and Exploratory Analysis Charts
  """
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  return

def preprocess_text(txt:str):
  """
  We are removing stopwords and punctuations and using the lemmatized
  tokens of the words
  """
    
  txt = re.sub('[^a-zA-Z]', ' ', txt)
  txt = txt.lower()
  txt = " ".join(txt.split())
  
  doc = nlp(txt)
  
  tokens_filtered = []
  
  for token in doc:
      if token.is_stop or token.is_punct:
          continue
          
      tokens_filtered.append(token.lemma_)
      
  return " ".join(tokens_filtered)


def vectorize_data(X_train, X_test):
  # We transform our texts into numbers.
  vectorizer = TfidfVectorizer()
  
  features_train = vectorizer.fit_transform(X_train)
  features_test = vectorizer.transform(X_test)
  
  features_train = features_train.toarray()
  features_test = features_test.toarray()
  return features_train, features_test


def initialize_models():
  """
  Initialize models
  """
  random_forest = RandomForestClassifier(random_state = SEED, n_jobs = -1)
  extra_trees = ExtraTreesClassifier(bootstrap = True, n_jobs = -1, random_state = SEED)
  xgb = XGBClassifier(random_state = SEED, n_jobs = -1)
  lgbm = LGBMClassifier(random_state = SEED, n_jobs = -1)
  cat_boost = CatBoostClassifier(random_state = SEED, verbose = 0)
  
  return [random_forest, extra_trees, xgb, lgbm, cat_boost]


if __name__=="__main__":
  create_figures_dir()
  df = load_data()
  
  # Create WordCloud before preprocessing.
  #===================================================================
  plots.create_word_cloud(df, os.path.join(figures_path, "word_cloud_before_preprocess.png"), "WordCloud Before PreProcessing Data")

  # Create Density Chart before preprocessing.
  plots.create_token_density_chart(df, os.path.join(figures_path, "density_chart_before_preprocess.png"), "Density Chart Before PreProcessing Data")

  
  df['Text'] = df['Text'].apply(preprocess_text)

  # Create WordCloud after preprocessing.
  #===================================================================
  plots.create_word_cloud(df, os.path.join(figures_path, "word_cloud_after_preprocess.png"), "WordCloud After PreProcessing Data")

  # Create Density Chart after preprocessing.
  plots.create_token_density_chart(df, os.path.join(figures_path, "density_chart_after_preprocess.png"), "Density Chart After PreProcessing Data")

  X = data['Text'].copy()
  y = data['Label']
  X_train, X_test, y_train, y_test = train_test_split(X, 
                                                      y, 
                                                      test_size = 0.3, 
                                                      random_state = SEED, 
                                                      stratify = y)
  print(f'X train total: {len(X_train)}')
  print(f'X test total: {len(X_test)}')

  print(" Training Class Value Counts: \n", y_train.value_counts())

  #Transform texts to vectors for fitting model
  X_train, X_test = vectorize_data(X_train, X_test)

  # Training
  # ----------------------------------------------------------------------------------------------------------------------------
  MODELS = initialize_models
  accuracy_train = {}
  accuracy_test = {}
  
  for model in tqdm(MODELS):
    name = type(model).__name__
    model.fit(features_train, y_train)
    y_pred_train = model.predict(features_train)
    accuracy_train[name] = accuracy_score(y_train, y_pred_train)
    y_pred_test = model.predict(features_test)
    accuracy_test[name] = accuracy_score(y_test, y_pred_test)
    print(f'* {name} finished')

  plots.create_metrics_figure(accuracy_train, accuracy_test, os.path.join(figures_path, "accuracy_train_and_test.png"))
  # You will see that the LGBM scores the best accuracy.
  # To train the LGBM Model again and get the confusion matrix, run final_model.py
    
  









  
