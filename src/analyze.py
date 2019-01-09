import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np

def analyze(pred_data, label):
    """
    Given the predicted label and the real label, it computes TN, FP, FN, TP, precision, recall, fscore, and accuracy
    """
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

    tn_perc = 0
    fp_prec = 0
    fn_prec = 0
    tp_prec = 0

    if fp + tn != 0:
        fp_prec = fp / (fp + tn)
        tn_prec = tn / (fp + tn)

    if fn + tp != 0:
        fn_prec = fn / (fn + tp)
        tp_prec = tp / (fn + tp)

    if tp + fp != 0:
        presicion = (tp) / (tp+fp)

    if tp + fn != 0:
        recall = (tp) / (tp+fn)

    if presicion + recall != 0:
        fscore = (2*presicion*recall) / (presicion + recall)

    accuracy = metrics.accuracy_score(pred_data, label)
    return [tn_prec, fp_prec, fn_prec, tp_prec, presicion, recall, accuracy, fscore]


def bargraph(tn, fp, fn, tp, precision, recall, acc, fscore):
    """
    Compares SGD, SVC, Naive Bayes methods by ploting out the True Positive %, False Positive %, False Negative %, True Positive %, Precision %, Recall %, Accuracy %, and Fscore %
    """
    n_groups = 3

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.9

    tn_rect = ax.bar(index, tn, bar_width, alpha=opacity, color='#d6c0b1', label='True Negative %')
    fp_rect = ax.bar(index + bar_width, fp, bar_width, alpha=opacity, color='#e5f0a8', label='False Positive %')
    fn_rect = ax.bar(index + bar_width * 2, fn, bar_width, alpha=opacity, color='#b8eca3', label='False Negative %')
    tp_rect = ax.bar(index + bar_width * 3, tp, bar_width, alpha=opacity, color='#f2b6c8', label='True Positive %')
    precision_rect = ax.bar(index + bar_width * 4, precision, bar_width, alpha=opacity, color='#b6eef2', label='Precision %')
    recall_rect = ax.bar(index + bar_width * 5, recall, bar_width, alpha=opacity, color='#b6ccf2', label='Recall %')
    acc_rect = ax.bar(index + bar_width * 6, acc, bar_width, alpha=opacity, color='#c4b6f2', label='Accuracy %')
    fscore_rect = ax.bar(index + bar_width * 7, fscore, bar_width, alpha=opacity, color='#efb6f2', label='Fscore %')

    plt.title('Comparative Analysis for SGD, SVC, Naive Bayes')
    plt.xlabel('Classification Methods')
    plt.ylabel('Percentage')
    plt.xticks(index + bar_width * 3.5, ('SGD', 'SVC', 'NB'))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.show()
