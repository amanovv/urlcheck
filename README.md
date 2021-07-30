# urlcheck

# Streamlit app to check whether certain news is fake or real based on Logistic Regression model

#### I used multiple featurizers to classify fake vs real news

When you first time load app, it will need to download couple of stuff including AI model file

Approaches are:

1. Domain name and keyword featurizer:

    - This is like searching domain names of the websites and counting certain keywords

2. Bag of Words featurizer:

    - Using one-hot-encoding method to encode website's description 
        - weaknesses: doesn't capture the word similarity and doesn't care about word order

3. Word Embeddings using GloVe:

    - Basically, creating word vectors for each word based on reference model that has been trained on large dataset
    - Captures word similarities pretty well

Eventually, I combine all features from above 3 approaches and train simple Logistic Regression model using scikit-learn

*Currently, trying to fix environment file issues on streamlit. It gives error for finding packages which I suspect because
I used mini-forge to manage my virtual envs since Mac M1 doesn't support conda-forge for now. 
I will give a try with my linux environment*

To use it now on your local host: 

    - Clone the repo
    - Install streamlit and other packages used/imported
    - run:
        > streamlit run streamlit_run.py

===================================================================================

Current work ongoing on:

    - Giving possiblity of training model from scratch by taking user specified hyperparemeters
    - Adding model selections, like neural networks, decision trees, random forest classifier

Future potential extension:

    - Making state-of-art model chrome extension
    - Creating executable or apk/ios probably
