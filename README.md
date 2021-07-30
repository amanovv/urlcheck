[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/amanovv/urlcheck)

# urlcheck

# Streamlit app to check whether certain news is fake or real using Machine Learning models

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

Eventually, I combine all features from above 3 approaches and train simple Logistic Regression model using scikit-learn.

The Results are:
- Logistic Regression
    - Testing accuracy of 79.9% 
    - Precision: 72.3% | lots of False Positives: Real news sometimes classified as Fake, probably some smaller scale news channels look like fake news site
    - Recall: 90.8%  | doing pretty well on False Negatives: Not classifying fake news as real which is good, at least not helping with misinformation lol
- Neural Networks:
    - Testing accuracy: 83.5%
    - Precision: 76% | doing better in False Positives
    - Recall: 92.9% | Really good

Neural networks usually give better results, that is expected. However, most companies prefer more explainable model which is going to be Logistic Regression. Neural networks are hard to explain since, they don't learn about features, I would say, they just transform data representation to more abstract hyperplane.

*Currently, trying to fix environment file issues on streamlit. It gives error for finding packages when trying to deploy it on streamlit server which I suspect it is because I used mini-forge to manage my virtual envs since Mac M1 doesn't support conda-forge for now. 
I will give a try with my linux environment*

*One more note if you are on Mac M1 and hence using mini-forge, you might have some bug on running some tensorflow features, gives memory error*

To use it now on your local host: 

    - Clone the repo
    - Install streamlit and other packages used/imported
    - run:
        > streamlit run streamlit_run.py

<img src="https://media.giphy.com/media/UDMnSA61SWKyHCq8RS/giphy.gif" width="600" height="500" />

===================================================================================

Current work ongoing on:

    - Giving possibility of training model from scratch by taking user specified hyperparemeters
    - Adding model selections, like neural networks, decision trees, random forest classifier

Future potential extension:

    - Training state-of-art model and making it chrome extension
    - Creating executable or apk/ios probably
