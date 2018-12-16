import string
import spacy
from analyze import *
from copy import deepcopy as DP

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords


from sklearn.cluster import KMeans
import metapy

import numpy as np

punctuations = string.punctuation
parser = spacy.load('en')

def parse_csv(file):
    """
    reads csv file data with labels and comment
    """
    f = open(file, 'r')
    import csv
    csv_reader = csv.reader(f, delimiter=',')
    data = []
    features = []
    labels = []
    label = None
    for line in csv_reader:
        comment = line[1]
        l = line[2]
        if "FALSE" in l:
            label = '-'
        elif "TRUE" in l:
            label = '+'
        elif "SPAM" in l:
            label = 'S'
        else:
            label = l
        data += [(comment, label)]
        features.append(DP(comment))
        labels.append(DP(label))

    return data, features, labels

def KMeansModel(features):
    """
    Uses Clustering(KMeans) instead of Classification to filter out fraud reivews.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    vectoriezed = vectorizer.fit_transform(features)

    num_cluster = 2
    kmean = KMeans(n_clusters = num_cluster, init = 'k-means++', max_iter = 100, n_init=1)
    kmean.fit(vectoriezed)

    # print 20 popular terms from k-means
    print("Top terms per cluster:")
    cluster_centers = kmean.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    for i in range(0, num_cluster):
        print("Cluster {}:".format(i)),
        for ind in cluster_centers[i, :20]:
            print(' {}'.format(terms[ind])),
        print()

    # test = vectorizer.transform(["This app is so bad"])
    # prediction = model.predict(test)
    # print(prediction)
    #
    # test = vectorizer.transform(["I love this app"])
    # prediction = model.predict(test)
    # print(prediction)

def SVCModel(data):
    """
    Uses Multinomial Support Vector Machine to train fraud review detector
    """
    #transformers using Spacy
    class predictors(TransformerMixin):
        def transform(self, X, **transform_params):
            return [self.clean_text(text) for text in X]
        def fit(self, X, y=None, **fit_params):
            return self
        def get_params(self, deep=True):
            return {}
        # cleans the text
        def clean_text(self, text):
            return text.strip().lower()

    #tokenizer to parse sentence, removing stopwords, removing punctuations and generate tokens using Spacy used for CountVectorizer()
    def spacy_tokenizer(sentence):
        tokens = parser(sentence)
        tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
        tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
        return tokens

    vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

    classifier = SVC(gamma='auto')

    # apply predictors(), vectorizer (transforms) and final estimator(classifier)
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', vectorizer),
                     ('classifier', classifier)])

    # Load sample data
    pipe.fit([x[0] for x in data], [x[1] for x in data])

    return pipe

def MultinomialNBModel(data):
    """
    Uses Multinomial Naive Bayes Classifer to train fraud review detector
    """
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words='english')

    # apply CountVectorizer(), TfidfTransformer(), (transforms) and final estimator(classifier)
    pipe = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', MultinomialNB())])

    # f = tfidf.fit_transform(np.array(features)).toarray()
    # mnb = snb.MultinomialNB()
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(features)
    # X_train_tfidf = TfidfTransformer().fit_transform(X_train_counts)

    pipe.fit([x[0] for x in data], [x[1] for x in data])
    return pipe

    # mnb.fit(X_train_counts, labels)
    # return mnb, count_vect

def SGDModel(data):
    """
    Uses Multinomial Stochastic Gradient Descent to train fraud review detector
    """
    classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=50, tol=1e-3)

    # apply CountVectorizer(), TfidfTransformer(), (transforms) and final estimator(classifier)
    pipe = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', classifier)])

    pipe.fit([x[0] for x in data], [x[1] for x in data])
    return pipe
