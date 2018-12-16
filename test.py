from train import *
from analyze import *
import argparse
from sklearn.metrics import accuracy_score


# training data set retrieved from https://github.com/amitness/applytics/blob/master/backend/training.cs

def runSVC(tests):
    """
    Wrapper function that uses training Support Vector Machine model to classify tests data
    """
    data, features, labels = parse_csv('training.csv')
    pipe = SVCModel(data)
    pred_data = pipe.predict([x[0] for x in tests])

    print ("SVC Accuracy:", accuracy_score([x[1] for x in tests], pred_data))
    return pred_data

def runMNB(tests):
    """
    Wrapper function that uses training Multinomial Naive Bayes model to classify tests data
    """
    data, features, labels = parse_csv('training.csv')
    pipe = MultinomialNBModel(data)
    pred_data = pipe.predict([x[0] for x in tests])

    print ("MNB Accuracy:", accuracy_score([x[1] for x in tests], pred_data))
    return pred_data

def runSGD(tests):
    """
    Wrapper function that uses training Stochastic Gradient Descent model to classify tests data
    """
    data, features, labels = parse_csv('training.csv')
    pipe = SGDModel(data)
    pred_data = pipe.predict([x[0] for x in tests])

    print ("SGD Accuracy:", accuracy_score([x[1] for x in tests], pred_data))
    return pred_data

def main():
    total = []
    data, features, labels = parse_csv('social.csv')
    print("Analysis for social apps")
    social_SGD_pred = runSGD(data)
    total.append(analyze(social_SGD_pred, labels))

    social_SVC_pred = runSVC(data)
    total.append(analyze(social_SVC_pred, labels))

    social_MNB_pred = runMNB(data)
    total.append(analyze(social_MNB_pred, labels))

    # each element in total looks like:
    # [tn, fp, fn, tp, precision, recall, acc, fscore]
    bargraph([i[0] for i in total], [i[1] for i in total], [i[2] for i in total], [i[3] for i in total], [i[4] for i in total], [i[5] for i in total], [i[6] for i in total], [i[7] for i in total])


    total = []
    data, features, labels = parse_csv('game.csv')
    print("Analysis for game apps")
    game_SGD_pred = runSGD(data)
    total.append(analyze(game_SGD_pred, labels))

    game_SVC_pred = runSVC(data)
    total.append(analyze(game_SVC_pred, labels))

    game_MNB_pred = runMNB(data)
    total.append(analyze(game_MNB_pred, labels))

    # [tn, fp, fn, tp, precision, recall, acc, fscore]
    bargraph([i[0] for i in total], [i[1] for i in total], [i[2] for i in total], [i[3] for i in total], [i[4] for i in total], [i[5] for i in total], [i[6] for i in total], [i[7] for i in total])

    total = []
    data, features, labels = parse_csv('education.csv')
    print("Analysis for education apps")
    education_SGD_pred = runSGD(data)
    total.append(analyze(education_SGD_pred, labels))

    education_SVC_pred = runSVC(data)
    total.append(analyze(education_SVC_pred, labels))

    education_MNB_pred = runMNB(data)
    total.append(analyze(education_MNB_pred, labels))

    # [tn, fp, fn, tp, precision, recall, acc, fscore]
    bargraph([i[0] for i in total], [i[1] for i in total], [i[2] for i in total], [i[3] for i in total], [i[4] for i in total], [i[5] for i in total], [i[6] for i in total], [i[7] for i in total])

main()
