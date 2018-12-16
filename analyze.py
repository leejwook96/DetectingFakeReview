import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def analyze(pred_data, label):
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    presicion = 0
    recall = 0
    fscore = 0
    for i in range(0, len(label)):
        pred = pred_data[i]
        real = label[i]
        if pred == '-':
            if real == '-':
                tn += 1
            else:
                fn += 1
        else:
            if real == '-':
                fp += 1
            else:
                tp += 1

    if tp + fp != 0:
        presicion = (tp) / (tp+fp)


    if tp + fn != 0:
        recall = (tp) / (tp+fn)

    if presicion + recall != 0:
        fscore = (2*presicion*recall) / (presicion + recall)


    accuracy = (tp + tn) / (tp + tn + fn + fp)
    print(tn, fp, fn, tp, presicion, recall, accuracy, fscore)

def drawgraph(tn, fp, fn, tp, precision, recall, acc, fscore):
    print(tn, fp, fn, tp, precision, recall, acc, fscore)
