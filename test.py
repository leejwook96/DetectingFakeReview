from train import *
from analyze import *
import argparse

# training data set retrieved from https://github.com/amitness/applytics/blob/master/backend/training.cs

def runSVC(tests):
    """
    """
    data, features, labels = parse_csv('training.csv')
    pipe = SVCModel(data)
    pred_data = pipe.predict([x[0] for x in tests])

    # print ("SVC Accuracy:", accuracy_score([x[1] for x in tests], pred_data))
    return pred_data

def runMNB(tests):
    """
    """
    data, features, labels = parse_csv('training.csv')
    pipe = MultinomialNBModel(data)
    pred_data = pipe.predict([x[0] for x in tests])

    # print ("MNB Accuracy:", accuracy_score([x[1] for x in tests], pred_data))
    return pred_data

def runSGD(tests):
    """
    """
    data, features, labels = parse_csv('training.csv')
    pipe = SGDModel(data)
    pred_data = pipe.predict([x[0] for x in tests])

    # print ("SGD Accuracy:", accuracy_score([x[1] for x in tests], pred_data))
    return pred_data

def main():
    data, features, labels = parse_csv('social.csv')
    print("Analysis for social apps")
    social_SVC_pred = runSVC(data)
    analyze(social_SVC_pred, labels)

    social_MNB_pred = runMNB(data)
    analyze(social_MNB_pred, labels)

    social_SGD_pred = runSGD(data)
    analyze(social_SGD_pred, labels)

    data, features, labels = parse_csv('game.csv')
    print("Analysis for game apps")
    game_SVC_pred = runSVC(data)
    analyze(game_SVC_pred, labels)

    game_MNB_pred = runMNB(data)
    analyze(game_MNB_pred, labels)

    game_SGD_pred = runSGD(data)
    analyze(game_SGD_pred, labels)

    data, features, labels = parse_csv('education.csv')
    print("Analysis for education apps")
    education_SVC_pred = runSVC(data)
    analyze(education_SVC_pred, labels)

    education_MNB_pred = runMNB(data)
    analyze(education_MNB_pred, labels)

    education_SGD_pred = runSGD(data)
    analyze(education_SGD_pred, labels)

main()
