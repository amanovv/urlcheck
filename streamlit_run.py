import streamlit as st
#import os
#import urllib
#import tensorflow as tf
#import pandas as pd 
#import cv2
#import time
import pickle
import wget
import requests, zipfile
import numpy as np
from featurizers.featurize import keyword_featurizer, vectorize_data_descriptions, glove_transform_data_descriptions
from bs4 import BeautifulSoup as bs
from torchtext.vocab import GloVe

def main():
  
  vec_data, loaded_model, glove = download_files(glove_vect_size=300)

  st.title("Fake news check")
  st.header("Paste the url of the news and you can check whether it is fake or real")
  
  
  st.sidebar.title("How you want to use")
  app_mode = st.sidebar.selectbox("Choose the modes here", 
  ["Use directly pre-trained", "Train from scratch"])
  if app_mode == "Use directly pre-trained":
    st.sidebar.success('provide the url for the news')

    news_url = st.text_input("Paste news url")

    url_ , html = get_data_pair(news_url)

    input_X = featurize_data_pair(url_,html,vec_data,glove,glove_vect_size=300)

    y_output = loaded_model.predict(input_X)[0]

    if y_output > 0.5:
      st.balloons()
      st.write("Url appears to be real news")
    else:
      st.write("ATTENTION, url appears to be fake news")

  elif app_mode == "Train from scratch":
    st.multiselect('Select with methods you want to use', 
    ['keyword featurizer', 'Bag of Words', 'GloVe word vectors'])
 

def get_description_from_html(html):
  soup = bs(html)
  description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'})
  if description_tag:
    description = description_tag.get('content') or ''
  else: # If there is no description, return empty string.
    description = ''
  return description


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
  
  bow_X = vectorize_data_descriptions([description],ref_data)
  
  # Approach 3.
  glove_X = glove_transform_data_descriptions([description], glove, glove_vect_size)
  
  X = combine_features([keyword_X, bow_X, glove_X])
  
  return X

@st.cache()
def download_files(glove_vect_size):
  # downloading file requires some work
  VEC_SIZE = glove_vect_size
  glove = GloVe(name='6B', dim=VEC_SIZE)
  finished = 0
  while finished == 0:
    st.spinner()
    data_url = "https://drive.google.com/file/d/1CSR2pyZfHCKqurdT5KYsseMiCfAblL_t/view?usp=sharing"

    url = "https://drive.google.com/file/d/1EK01cw3cEe8ikDk2PZ8O3PAxyCj0HYmw/view?usp=sharing"
    filename = wget.download(url)
    loaded_model = pickle.load(open(filename, 'rb'))
    finished = 1
    datafile = wget.download(data_url)
    with zipfile.ZipFile(datafile,'r') as zipObj:
      zipObj.extractall()
    

    train_data, val_data = pickle.load(train_val_data.pkl)

  return train_data, loaded_model, glove


if __name__ == "__main__":
    main()


'''


#filename = tf.keras.utils.get_file("emotion_detection_model_for_streamlit.h5", url)
EMOTIONS = ['ANGRY', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']

st.title("Emotion Detector")
st.header("This app detects your emotions! upload a picture to try it out!")

model = tf.keras.models.load_model("emotion_detection_model_for_streamlit.h5")
#model_lm = tf.keras.models.load_model("best_lm_model.h5")
f = st.file_uploader("Upload Image")

if f is not None: 
  # Extract and display the image
  file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  st.image(image, channels="BGR")

  # Prepare the image
  resized = cv2.resize(image, (48, 48), interpolation=cv2.INTER_LANCZOS4)
  gray_1d = np.mean(resized, axis=-1)
  gray = np.zeros_like(resized)
  gray[:,:,0] = gray_1d
  gray[:,:,1] = gray_1d
  gray[:,:,2] = gray_1d
  normalized = gray/255
  model_input = np.expand_dims(normalized,0)

  # Run the model
  scores_transfer = model.predict(model_input)
  #scores_lm_model = model_lm.predict(model_input.reshape())

  with st.spinner(text='predicting ...'):
      time.sleep(5)
      st.success('Prediction done')

  st.balloons()
  # Print results and plot score
  st.write(f"The predicted emotion with transfer learning is: {EMOTIONS[scores_transfer.argmax()]}")


  df = pd.DataFrame(scores_transfer.flatten(), columns = EMOTIONS)
  #df["Emotion"] = EMOTIONS
  #df["Scores_transfer"] = scores_transfer.flatten()


  st.area_chart(df)
  st.balloons()

  
'''