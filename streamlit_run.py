import streamlit as st
import os
#import urllib
#import tensorflow as tf
#import pandas as pd 
#import cv2
#import time
import pickle
import wget

import requests, zipfile
import numpy as np
from featurizers.featurize import keyword_featurizer, vectorize_data_descriptions, glove_transform_data_descriptions,get_description_from_html

from torchtext.vocab import GloVe

def main():
  vec_data, loaded_model = download_model()
  glove = download_files(glove_vect_size=300)
  st.title("Fake news check")
  st.header("Paste the url of the news and you can check whether it is fake or real")
  
  
  st.sidebar.title("How you want to use")
  app_mode = st.sidebar.selectbox("Choose the modes here", 
  ["Use directly pre-trained", "Train from scratch"])
  if app_mode == "Use directly pre-trained":
    st.sidebar.success('provide the url for the news')

    news_url = st.text_input("Paste news url")

    url, html = get_data_pair(news_url)

    input_X = featurize_data_pair(url,html,vec_data,glove,glove_vect_size=300)

    y_output = loaded_model.predict(input_X)[0]

    if y_output > 0.5:
      st.balloons()
      st.write("Url appears to be real news")
    else:
      st.write("ATTENTION, url appears to be fake news")

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


def combine_features(X_list):
  return np.concatenate(X_list, axis=1)

def dict_to_features(features_dict):
  X = np.array(list(features_dict.values())).astype('float')
  X = X[np.newaxis, :]
  return X
def featurize_data_pair(url, html,ref_data,glove, glove_vect_size):
  # domain check and keywords count features
  keyword_X = dict_to_features(keyword_featurizer(url, html))
  # Approach 2.
  description = get_description_from_html(html)
  
  bow_X = vectorize_data_descriptions([description],ref_data,n_features=300)
  
  # Approach 3.
  glove_X = glove_transform_data_descriptions([description], glove, glove_vect_size)
  
  X = combine_features([keyword_X, bow_X, glove_X])
  
  return X



@st.cache
def download_model():
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

  return train_data, loaded_model


#@st.cache()
def download_files(glove_vect_size):
  # downloading file requires some work
  VEC_SIZE = glove_vect_size
  glove = GloVe(name='6B', dim=VEC_SIZE)
  #finished = 0
  #while finished == 0:
    #st.spinner()

    #os.system(f"gdown https://drive.google.com/file/d/1CSR2pyZfHCKqurdT5KYsseMiCfAblL_t/view?usp=sharing -O data.zip")

    #os.system(f"gdown https://drive.google.com/file/d/1W6iQHmge5bhYjEpi2bwuwhj0lK3BMMhk/view?usp=sharing -O LR_model.pkl")
  return glove


if __name__ == "__main__":
    main()