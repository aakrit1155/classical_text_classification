import os
import joblib
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

# Model
#===========================================================================
from lightgbm import LGBMClassifier

# Metrics
#===========================================================================
from sklearn.metrics import confusion_matrix, accuracy_score

# Data visualization
#===========================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# tqdm
#===========================================================================
from tqdm.auto import tqdm
from . import plots

nlp = spacy.load("en_core_web_sm")

SEED = 43
file_path = "./data/df_file.csv"
figures_path = "./figures"
model_save_path = "./saved_model"



def create_model_dir(model_path=model_save_path):
  """
  Create Directory to save the Best Model: LGBM
  """
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  return

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
  return features_train, features_test, vectorizer


def initialize_model():
  """
  Initialize model
  """
  lgbm = LGBMClassifier(random_state = SEED, n_jobs = -1)
  
  return lgbm


if __name__=="__main__":
  create_model_dir()
  df = load_data()

  
  df['Text'] = df['Text'].apply(preprocess_text)


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
  X_train, X_test, TFIDF_vectorizer = vectorize_data(X_train, X_test)

  # Training
  # ----------------------------------------------------------------------------------------------------------------------------
  MODEL = initialize_model()
  MODEL.fit(X_train, y_train)

  y_pred_train = MODEL.predict(X_train)
  y_pred_test = MODEL.predict(X_test)
  
  # Save the TF-IDF vectorizer
  joblib.dump(TFIDF_vectorizer, os.path.join(model_save_path,'tfidf_vectorizer.pkl'))
  
  # Save the LightGBM model
  joblib.dump(MODEL, os.path.join(model_save_path,'lgbm_model.pkl'))
  
  cf_mx_train = confusion_matrix(y_train, y_pred_train)
  cf_mx_test = confusion_matrix(y_test, y_pred_test)
  
  fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,5))
  
  sns.heatmap(cf_mx_train, cmap = 'Reds', 
              fmt = '', annot = True, cbar = False, 
              annot_kws = {'fontsize':10, 'fontweight':'bold'},
              square = False, ax = ax[0])
  
  sns.heatmap(cf_mx_test, cmap = 'Blues', 
              fmt = '', annot = True, cbar = False, 
              annot_kws = {'fontsize':10, 'fontweight':'bold'},
              square = False, ax = ax[1])
  
  ax[0].set_title("Confusion Matrix Train", fontsize = 12, fontweight = "bold", color = "black")
  ax[1].set_title("Confusion Matrix Test", fontsize = 12, fontweight = "bold", color = "black")
  fig.tight_layout()
  fig.savefig(os.path.join(figures_path, "Confusion_Matrix_LGBM.png"))

  """
  To use the model in new data, you will have to load both tfidf_vectorizer, and 
  the model file using the following code:
  
  # Load the TF-IDF vectorizer
  loaded_tfidf = joblib.load( os.path.join(model_save_path,'tfidf_vectorizer.pkl'))
  
  # Load the LightGBM model
  loaded_model = joblib.load( os.path.join(model_save_path,'lgbm_model.pkl'))

  # Use the loaded_model.predict on the preprocessed and vectorized text like below:
  sample_text="This is a sample text used for model prediction.."
  preprocessed_text=preprocess_text(sample_text)
  feature_text = loaded_tfidf.transform(preprocessed_text).toarray()
  loaded_model.predict(feature_text)

  You will get the output as an integer from 0 to 4 and they mean the following:
  Politics = 0
  Sport = 1
  Technology = 2
  Entertainment =3
  Business = 4
  """
  
  
