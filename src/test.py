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

def test(filename, graph):
    total = []
    data, features, labels = parse_csv(filename)

    SGD_pred = runSGD(data)
    total.append(analyze(SGD_pred, labels))

    SVC_pred = runSVC(data)
    total.append(analyze(SVC_pred, labels))

    MNB_pred = runMNB(data)
    total.append(analyze(MNB_pred, labels))

    # each element in total looks like:
    # [tn, fp, fn, tp, precision, recall, acc, fscore]
    if (graph):
        bargraph([i[0] for i in total], [i[1] for i in total], [i[2] for i in total], [i[3] for i in total], [i[4] for i in total], [i[5] for i in total], [i[6] for i in total], [i[7] for i in total])

def main():
    print("Analysis for social apps")
    test('social.csv', graph=True)

    print("Analysis for game apps")
    test('game.csv', graph=True)

    print("Analysis for educatoin apps")
    test('education.csv', graph=True)

if __name__ == '__main__':
    main()
