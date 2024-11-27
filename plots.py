# Data visualization
#===========================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS


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
  
