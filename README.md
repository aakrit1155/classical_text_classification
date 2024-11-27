# classical_text_classification
Text Classification Using Classical ML algorithms like RandomForestClassifier, ExtraTreesClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier

## The dataset used in the project is from Kaggle 
# link to the dataset:
<a href="https://www.kaggle.com/datasets/tanishqdublish/text-classification-documentation">Dataset</a>

# The data contains news_articles that are classified among 5 different categories:
- Politics = 0
- Sport = 1
- Technology = 2
- Entertainment =3
- Business = 4

# Steps to follow:
- Create a virtual environment in python. Preferably python 3.10 and activate the environment
- install the dependencies using 'pip install -r requirements.txt'
- run the models.py file using 'python models.py' This trains and tests on the various models mentioned above and you will find the figures on 'figures' folder/directory
- Run the final_model.py using 'python final_model.py'. Here we train the best performing model from the previous run and save the model file
- Follow the final comments in 'final_model.py' file to use the model for inference on news texts to classify the texts
