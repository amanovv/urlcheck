import requests, zipfile
import streamlit as  st
from sklearn.feature_extraction.text import CountVectorizer
import math
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
######### DOMAIN NAME and KEYWORD featurizer ################

# Gets the log count of a phrase/keyword in HTML (transforming the phrase/keyword
# to lowercase).
def get_normalized_count(html, phrase):
    return math.log(1 + html.count(phrase.lower()))

# Returns a dictionary mapping from plaintext feature descriptions to numerical
# features for a (url, html) pair.
def keyword_featurizer(url, html):
    features = {}
    
    #features['.com domain'] = url.endswith('.com') .com domain does't add much information for classifying
    features['.org domain'] = url.endswith('.org')
    features['.net domain'] = url.endswith('.net')
    features['.info domain'] = url.endswith('.info')
    features['.org domain'] = url.endswith('.org')
    features['.biz domain'] = url.endswith('.biz')
    features['.ru domain'] = url.endswith('.ru')
    features['.co.uk domain'] = url.endswith('.co.uk')
    features['.co domain'] = url.endswith('.co')
    features['.tv domain'] = url.endswith('.tv')
    features['.news domain'] = url.endswith('.news')
    
    keywords = ['trump', 'biden', 'clinton', 'sports', 'finance'] # as fake news landscape changes can add more key words
    
    for keyword in keywords:
      features[keyword + ' keyword'] = get_normalized_count(html, keyword)
    
    return features

############################################################################################################



######### Bag of Words featurizer ################
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

def vectorize_data_descriptions(data_descriptions, ref_data, n_features =300):
  vectorizer = CountVectorizer(max_features=n_features)
  ref_descriptions = get_descriptions_from_data(ref_data)
  vectorizer.fit(ref_descriptions)
  X = vectorizer.transform(data_descriptions).todense()
  return X


############################################################################################################



######### GLoVe Word Embeddings featurizer ################

# Returns word vector for word if it exists, else return None.
def get_word_vector(word,glove):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

def glove_transform_data_descriptions(descriptions,glove, vect_size):
    X = np.zeros((len(descriptions), vect_size))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split(): 
            vec = get_word_vector(word,glove)
            if vec is not None:

                found_words += 1
                X[i] += vec
                
        # We divide the sum by the number of words added, so we have the
        # average word vector.
        if found_words > 0:
            X[i] /= found_words
            
    return X


'''
def prepare_data(data, featurizer):
    X = []
    y = []
    for datapoint in data:
        url, html, label = datapoint
        # We convert all text in HTML to lowercase, so <p>Hello.</p> is mapped to
        # <p>hello</p>. This will help us later when we extract features from 
        # the HTML, as we will be able to rely on the HTML being lowercase.
        html = html.lower() 
        y.append(label)

        features = featurizer(url, html)

        
        feature_descriptions, feature_values = zip(*features.items())

        X.append(feature_values)

    return X, y, feature_descriptions
'''