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


file_path = "../data/df_file.csv"
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
  Create Direectory to save plots and Exploratory Analysis Charts
  """
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  return









  
