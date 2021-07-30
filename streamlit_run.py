import streamlit as st
import os
import pickle
import wget
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
import requests, zipfile
import numpy as np
from featurizers.featurize import keyword_featurizer, vectorize_data_descriptions, glove_transform_data_descriptions
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.vocab import GloVe

def main():

  st.write("Downloading some usefull stuff, it may take some time ~ 5 mins")
  vectorizer, loaded_model = download_model()
  glove = download_files(glove_vect_size=300)
  st.title("Fake news check")
  st.header("Paste the url of the news and you can check whether it is fake or real")
  
  
  st.sidebar.title("How you want to use")
  app_mode = st.sidebar.selectbox("Choose the modes here", 
  ["Use directly pre-trained", "Train from scratch"])
  if app_mode == "Use directly pre-trained":
    st.sidebar.success('provide the url for the news')


    news_url = st.text_input("Paste news url")
    button = st.button('Summon AI fact checker')
    st.spinner()
    if button:
      url, html = get_data_pair(news_url)

      input_X = featurize_data_pair(url,html,vectorizer,glove,glove_vect_size=300)

      y_output = loaded_model.predict(input_X)[0]

      if y_output < 0.5:
        st.balloons()
        st.write("Url appears to be real news")
      else:
        st.write("ATTENTION, url appears to be fake news")
        st.warning("Please check the news source and also reporter's bio")

  elif app_mode == "Train from scratch":
    st.multiselect('Select with methods you want to use', 
    ['keyword featurizer', 'Bag of Words', 'GloVe word vectors'])



def get_data_pair(url='https://finance.yahoo.com'):
  if not url.startswith('http'):
      url = 'http://' + url
  url_pretty = url
  if url_pretty.startswith('http://'):
      url_pretty = url_pretty[7:]
  if url_pretty.startswith('https://'):
      url_pretty = url_pretty[8:]
      
  # Scrape website for HTML
  response = requests.get(url, timeout=10)
  htmltext = response.text
  return url_pretty, htmltext

def get_description_from_html(html):
  soup = bs(html)
  description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'})
  if description_tag:
    description = description_tag.get('content') or ''
  else: # If there is no description, return empty string.
    description = ''
  return description

def get_descriptions_from_data(data):
  # A dictionary mapping from url to description for the websites in 
  # train_data.
  descriptions = []
  for site in tqdm(data):
    url, html, label = site
    descriptions.append(get_description_from_html(html))
  return descriptions

def combine_features(X_list):
  return np.concatenate(X_list, axis=1)

def dict_to_features(features_dict):
  X = np.array(list(features_dict.values())).astype('float')
  X = X[np.newaxis, :]
  return X
def featurize_data_pair(url, html,vectorizer,glove, glove_vect_size):
  # domain check and keywords count features
  keyword_X = dict_to_features(keyword_featurizer(url, html))
  # Approach 2.
  description = get_description_from_html(html)
  
  bow_X = vectorize_data_descriptions([description],vectorizer)
  
  # Approach 3.
  glove_X = glove_transform_data_descriptions([description], glove, glove_vect_size)
  
  X = combine_features([keyword_X, bow_X, glove_X])
  
  return X



@st.cache
def download_model():

  # Downloading stuff from google drive using wget sucks
  # Dropbox is the solution, I figured

  data_url = 'https://drive.google.com/file/d/1CSR2pyZfHCKqurdT5KYsseMiCfAblL_t/view?usp=sharing'

  data_url_dropbox = 'https://www.dropbox.com/s/5jqlxaoxf3hlsrh/newsdata.zip?dl=0'

  url_dropbox = 'https://www.dropbox.com/s/f2m04sl0gn4140f/LR_model.pkl?dl=0'

  url = 'https://drive.google.com/file/d/1W6iQHmge5bhYjEpi2bwuwhj0lK3BMMhk/view?usp=sharing'

  os.system(f"wget -O data.zip 'https://www.dropbox.com/s/5jqlxaoxf3hlsrh/newsdata.zip?dl=0' ")
  os.system(f"wget -O LR_model.pkl 'https://www.dropbox.com/s/f2m04sl0gn4140f/LR_model.pkl?dl=0' ")

  #output = "LR_model.pkl"
  #output1 = 'data.zip'
  #gdown.download(url,output)
  #gdown.download(data_url,output1)

  with open("LR_model.pkl", 'rb') as file:  
    loaded_model = pickle.load(file)
  #loaded_model = pickle.load(open('LR_model.sav','rb'))

  with zipfile.ZipFile('data.zip','r') as zipObj:
    zipObj.extractall()
  
  with open('train_val_data.pkl', 'rb') as f:
    train_data, val_data = pickle.load(f)


  vectorizer = CountVectorizer(max_features=300)
  ref_descriptions = get_descriptions_from_data(val_data)
  vectorizer.fit(ref_descriptions)

  return vectorizer, loaded_model


#@st.cache()
def download_files(glove_vect_size):
  # downloading file requires some work
  VEC_SIZE = glove_vect_size
  glove = GloVe(name='6B', dim=VEC_SIZE)
  return glove


if __name__ == "__main__":
    main()
