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
from keras.models import load_model
import tensorflow.keras
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape, Dense, Conv2D

def main():
  
  vectorizer_lr, vectorizer_nn, loaded_model_lr, loaded_model_nn = download_model()
  glove = download_files(glove_vect_size=300)


  st.sidebar.title("What you want to use")
  app_mode = st.sidebar.selectbox("Choose the modes here", 
  ["Use directly pre-trained", "Train from scratch"])
  
  
  #st.spinner("Downloading some usefull stuff, it may take some time ~ 5 mins")
  
  

  
  
  if app_mode == "Use directly pre-trained":
    st.title("Fake news check")
    st.header("Paste the url of the news and you can check whether it is fake or real")
    st.sidebar.success('provide the url for the news')
    #selection = st.selectbox('Select which model to use for classification',['Neural Network','Logistic Regression'])
    #if selection == 'Neural Network':
      

      
    
    
    #elif selection == 'Logistic Regression':
      #, loaded_model_lr = download_model(max_features = 300)

    news_url = st.text_input("Paste news url")
    button = st.button('Summon AI fact checker')
    
    if button:
      url, html = get_data_pair(news_url)
      #st.subheader("Alright, you are using Logistic Regression")

      input_X, input_X_nn = featurize_data_pair(url,html,vectorizer_lr,vectorizer_nn,glove,glove_vect_size=300)
      y_output = loaded_model_lr.predict(input_X)[0]
      probs=loaded_model_lr.predict_proba(input_X)
      output_nn = loaded_model_nn.predict(np.array(input_X_nn))
      prediction = loaded_model_nn.predict_classes(np.array(input_X_nn))

      
      st.balloons()
      st.write("Results with Logistic Regression")

      st.success("REALNESS percentage: " + (str(round(probs[0][0]*100, 2)) + "%" + "real"))
      #st.write("URL appears to be: ", str(round(probs[0][0]*100, 2)) + "%", "real")
      st.error("FAKENESS percentage: " + (str(round(probs[0][1]*100, 2)) + "%" + "fake"))
      #st.write("URL appears to be: ", str(round(probs[0][1]*100, 2)) + "%", "fake")
      if y_output < 0.5:
        st.write("URL appears to be real news")
      else:
        st.write("ATTENTION, URL appears to be fake news")
        st.warning("Please check the news source and also reporter's bio")

      st.write("Results with Neural Network")  
      #st.subheader("Oooooo, you are using neural networks")
      #input_X_nn = featurize_data_pair(url,html,vectorizer_nn,glove,glove_vect_size=300)
      
      
      st.success("REALNESS percentage: " + (str(round(output_nn[0][0]*100, 2)) + "%", "real"))
      #st.write("URL appears to be: ", )
      st.error("FAKENESS percentage: " + (str(round(output_nn[0][1]*100, 2)) + "%", "fake"))
      #st.write("URL appears to be: ", str(round(probs[0][1]*100, 2)) + "%", "fake")
      
      if prediction[0] < 0.5:
        st.write("URL appears to be real news")
      else:
        st.write("ATTENTION, URL appears to be fake news")
        st.warning("Please check the news source and also reporter's bio")

      st.write("Here is Neural Network architecture details that you are using")
      st.write(loaded_model_nn.summary())
      

  elif app_mode == "Train from scratch":
    st.multiselect('Select with methods you want to use', 
    ['keyword featurizer', 'Bag of Words', 'GloVe word vectors'])



def get_data_pair(url):
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

def featurize_data_pair(url, html,vectorizer, vectorizer_nn, glove, glove_vect_size):
  # domain check and keywords count features
  keyword_X = dict_to_features(keyword_featurizer(url, html))
  # bag of words
  description = get_description_from_html(html)
  
  bow_X = vectorize_data_descriptions([description],vectorizer)
  bow_X_nn = vectorize_data_descriptions([description],vectorizer_nn)
  
  # glove
  glove_X = glove_transform_data_descriptions([description], glove, glove_vect_size)
  
  X = combine_features([keyword_X, bow_X, glove_X])
  X_nn = combine_features([keyword_X, bow_X_nn, glove_X])
  
  return X, X_nn



@st.cache(allow_output_mutation=True)
def download_model():

  # Downloading stuff from google drive using wget sucks
  # Dropbox is the solution, I figured

  data_url = 'https://drive.google.com/file/d/1CSR2pyZfHCKqurdT5KYsseMiCfAblL_t/view?usp=sharing'

  data_url_dropbox = 'https://www.dropbox.com/s/5jqlxaoxf3hlsrh/newsdata.zip?dl=0'

  url_dropbox = 'https://www.dropbox.com/s/f2m04sl0gn4140f/LR_model.pkl?dl=0'

  url = 'https://drive.google.com/file/d/1W6iQHmge5bhYjEpi2bwuwhj0lK3BMMhk/view?usp=sharing'

  os.system(f"wget -O data.zip 'https://www.dropbox.com/s/5jqlxaoxf3hlsrh/newsdata.zip?dl=0' ")
  os.system(f"wget -O LR_model.pkl 'https://www.dropbox.com/s/f2m04sl0gn4140f/LR_model.pkl?dl=0' ")

  #neural network download
  os.system(f"wget -O model.h5 'https://www.dropbox.com/s/npdk4yzqoi69362/cnn_model.h5?dl=0' ")
  #os.system(f"wget -O model.json 'https://www.dropbox.com/s/kerf3fljhvsxpq9/nn_model.json?dl=0' ")
  #json_file = open('model.json', 'r')
  #loaded_model_json = json_file.read()
  #json_file.close()
  #loaded_model_nn = model_from_json(loaded_model_json)
  #loaded_model_nn.load_weights("model.h5")
  #loaded_model_nn._make_predict_function()
  loaded_model_nn = load_model("model.h5")

  

  

  
  with zipfile.ZipFile('data.zip','r') as zipObj:
    zipObj.extractall()
  
  with open('train_val_data.pkl', 'rb') as f:
    train_data, val_data = pickle.load(f)
  #output = "LR_model.pkl"
  #output1 = 'data.zip'
  #gdown.download(url,output)
  #gdown.download(data_url,output1)
  with open("LR_model.pkl", 'rb') as file:  
      loaded_model = pickle.load(file)
  ref_descriptions = get_descriptions_from_data(val_data)
  #if max_features == 300:
  vectorizer_lr = CountVectorizer(max_features=300)
  #elif max_features == 685:
  vectorizer_nn = CountVectorizer(max_features=685)
    #json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    
  #loaded_model = pickle.load(open('LR_model.sav','rb'))

  vectorizer_lr.fit(ref_descriptions)
  vectorizer_nn.fit(ref_descriptions)

  return vectorizer_lr, vectorizer_nn, loaded_model, loaded_model_nn


#@st.cache()
def download_files(glove_vect_size):
  # downloading file requires some work
  VEC_SIZE = glove_vect_size
  glove = GloVe(name='6B', dim=VEC_SIZE)
  return glove


if __name__ == "__main__":
  main()
