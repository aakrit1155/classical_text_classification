# Data visualization
#===========================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import pandas as pd


def create_word_cloud(data, save_path, title):
  all_text = " ".join(data['Text'])

  wc = WordCloud(stopwords = STOPWORDS).generate(all_text)
  
  plt.figure(figsize = (20,10))
  plt.imshow(wc)
  plt.axis('off')
  plt.title(title)
  plt.savefig(save_path)


def create_token_density_chart(data, save_path, title):
  seq_len = []

  for txt in data.Text:
      seq_len.append(len(txt.split()))
      
  print(f'Max Sequence in the data: {max(seq_len)}')
  plt.figure(figsize = (20,10))
  sns.histplot(seq_len, kde = True, line_kws = {'linewidth': 2.3})
  sns.rugplot(seq_len)
  plt.title(title)
  plt.savefig(save_path)


def create_metrics_figure(metric_train, metric_test, save_path):
  metric_train = pd.DataFrame.from_dict(metric_train, orient = 'index')
  metric_train = metric_train.rename(columns = {0:'Train'})
  
  metric_test = pd.DataFrame.from_dict(metric_test, orient = 'index')
  metric_test = metric_test.rename(columns = {0:'Test'})
  
  fig,ax = plt.subplots(figsize = (12,5))
  plt.style.use('ggplot')
  
  labels = metric_train.index.to_list()
  values_train = metric_train.iloc[:,0].to_list()
  values_test = metric_test.iloc[:,0].to_list()
  x = np.arange(len(labels))
  width = 0.35
  
  rects1 = ax.bar(x = x - width/2, height = values_train, width = width)
  rects2 = ax.bar(x = x + width/2, height = values_test, width = width)

  def autolabel(rects):
    for rect in rects:
      height = rect.get_height()
      ax.annotate(text = f'{height:.4f}', 
                  xy = (rect.get_x() + rect.get_width()/2, height), 
                  xytext = (0,3), 
                  textcoords = "offset points", 
                  ha = "center", 
                  va = "bottom")

  autolabel(rects1)
  autolabel(rects2)
  ax.set_title("Metric of Performance: Accuracy", fontsize = 12, fontweight = "bold", color = "black")
  ax.set_ylabel("score", fontsize = 8, fontweight = "bold", color = "black")
  ax.set_xlabel("Models", fontsize = 8, fontweight = "bold", color = "black")
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  fig.savefig(save_path)

    
