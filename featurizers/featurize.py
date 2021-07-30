import requests, zipfile
import streamlit as  st
from sklearn.feature_extraction.text import CountVectorizer
import math
import numpy as np

######### DOMAIN NAME and KEYWORD featurizer ################

# Gets the log count of a phrase/keyword in HTML (transforming the phrase/keyword
# to lowercase).
def get_normalized_count(html, phrase):
    return math.log(1 + html.count(phrase.lower()))

# Returns a dictionary mapping from plaintext feature descriptions to numerical
# features for a (url, html) pair.
def keyword_featurizer(url, html):
    features = {}
    
    features['.com domain'] = url.endswith('.com') #.com domain does't add much information for classifying
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


def vectorize_data_descriptions(data_descriptions, vectorizer):
  
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
                
        # divide the sum by the number of words added, so then have the
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
        # convert all text in HTML to lowercase, so <p>Hello.</p> is mapped to
        # <p>hello</p>. This will help later when extracting features from 
        # the HTML, as will be able to rely on the HTML being lowercase.
        html = html.lower() 
        y.append(label)

        features = featurizer(url, html)

        
        feature_descriptions, feature_values = zip(*features.items())

        X.append(feature_values)

    return X, y, feature_descriptions
'''